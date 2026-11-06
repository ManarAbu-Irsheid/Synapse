import numpy as np

def label_frames(df, thr=0.25, min_s=1.5, rate=5, margin_s=2.0, neg_ratio=0.6):
    """يرجع (labels, segments). label: 1=تثاؤب، 0=مش تثاؤب، -1=تجاهل.
    rate = عدد الصفوف بالثانية (25 fps / step 5 = 5)."""
    m = df.mar.rolling(5, center=True, min_periods=1).median().values
    face = df.face.values.astype(bool)
    on = (np.nan_to_num(m) > thr) & face

    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) / rate >= min_s:
                segs.append((i, j))
            i = j
        else:
            i += 1

    lab = np.full(len(df), -1)
    idx = np.arange(len(df))
    dist = np.full(len(df), np.inf)
    for a, b in segs:
        lab[a:b] = 1
        d = np.where(idx < a, a - idx, np.where(idx >= b, idx - b + 1, 0))
        dist = np.minimum(dist, d)
    neg = face & (dist > margin_s * rate) & (np.nan_to_num(m) < thr * neg_ratio)
    lab[neg & (lab != 1)] = 0
    return lab, segs
