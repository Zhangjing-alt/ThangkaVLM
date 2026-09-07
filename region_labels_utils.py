# -*- coding: utf-8 -*-
"""
在 merger 输出的 token 网格上构建 region_labels。

关键事实（Qwen2-VL / Qwen3-VL 系列）：
  image_processor 输出 pixel_values 的 patch 顺序已按 merge block 分组，
  即 (t, Hm, ms, Wm, ms) 展平，因此 merger 之后 token 顺序 = 合并网格行主序：
      idx = ti*(Hm*Wm) + r*Wm + c
  这一点必须用 sanity_check_region_labels() 实测确认后才可依赖。
"""
import numpy as np
import torch


def normalize_boxes(elements, image_wh):
    """elements: [{'bbox':[x1,y1,x2,y2], ...}]  -> [[x1n,y1n,x2n,y2n], ...]"""
    W, H = image_wh
    out = []
    for el in elements:
        bb = el.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        if max(x1, y1, x2, y2) > 1.5:          # 像素坐标
            x1, x2 = x1 / W, x2 / W
            y1, y2 = y1 / H, y2 / H
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2])
    return out


def build_region_labels_on_merged_grid(boxes_norm, grid_thw, merge_size=2,
                                       min_cells=1):
    """
    boxes_norm : [K,4] 归一化 xyxy，region id = 1..K
    grid_thw   : (t,h,w)，来自 image_grid_thw（merger 之前）
    return     : LongTensor [N]，N = t*(h//ms)*(w//ms)
    """
    t, h, w = [int(v) for v in torch.as_tensor(grid_thw).reshape(-1)[:3]]
    ms = int(merge_size)
    Hm, Wm = h // ms, w // ms
    N = t * Hm * Wm
    labels = torch.zeros(N, dtype=torch.long)
    if not boxes_norm:
        return labels

    # 大框先画、小框后画 -> 重叠时小区域覆盖大区域（保留细粒度元素）
    order = sorted(range(len(boxes_norm)),
                   key=lambda i: -((boxes_norm[i][2] - boxes_norm[i][0]) *
                                   (boxes_norm[i][3] - boxes_norm[i][1])))
    for i in order:
        x1, y1, x2, y2 = boxes_norm[i]
        c1 = int(np.floor(x1 * Wm)); c2 = int(np.ceil(x2 * Wm))
        r1 = int(np.floor(y1 * Hm)); r2 = int(np.ceil(y2 * Hm))
        c1 = max(0, min(c1, Wm - 1)); r1 = max(0, min(r1, Hm - 1))
        c2 = max(c1 + min_cells, c2);  r2 = max(r1 + min_cells, r2)   # 极小框保底
        c2 = min(c2, Wm);              r2 = min(r2, Hm)
        for ti in range(t):
            base = ti * Hm * Wm
            for r in range(r1, r2):
                labels[base + r * Wm + c1: base + r * Wm + c2] = i + 1
    return labels


def sanity_check_region_labels(grid_thw, merge_size=2, verbose=True):
    """左上 1/4 框应只命中左上区域；若不成立说明 token 顺序不是行主序。"""
    t, h, w = [int(v) for v in torch.as_tensor(grid_thw).reshape(-1)[:3]]
    ms = int(merge_size); Hm, Wm = h // ms, w // ms
    lab = build_region_labels_on_merged_grid([[0., 0., .5, .5]], grid_thw, ms)
    m = lab[:Hm * Wm].view(Hm, Wm).float()
    tl = m[:Hm // 2, :Wm // 2].mean().item()
    br = m[Hm // 2:, Wm // 2:].mean().item()
    cov = (lab > 0).float().mean().item()
    if verbose:
        print(f"[sanity] merged grid=({Hm},{Wm})  N={lab.numel()}")
        print(f"[sanity] 左上命中率={tl:.3f} (期望≈1.0)")
        print(f"[sanity] 右下命中率={br:.3f} (期望≈0.0)")
        print(f"[sanity] 总覆盖率={cov:.3f} (期望≈0.25)")
    return tl > 0.95 and br < 0.05


def region_labels_stats(labels):
    lab = labels.reshape(-1)
    K = int(lab.max().item())
    fg = (lab > 0).float().mean().item()
    sizes = [int((lab == r).sum().item()) for r in range(1, K + 1)]
    return {"N": lab.numel(), "num_regions": K,
            "fg_ratio": round(fg, 4), "region_sizes": sizes,
            "empty_regions": sum(1 for s in sizes if s == 0)}
