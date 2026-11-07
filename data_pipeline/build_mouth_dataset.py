"""Build the mouth (yawn / no_yawn) dataset from NITYMED, from scratch.

Usage:
  python build_mouth_dataset.py --root <.../DSM_Dataset-HDTV720> --out <output dir>

Steps per Yawning video: extract EAR/MAR/head-pose landmarks, label yawn frames
(labeling.label_frames), keep videos with exactly 3 yawn segments, crop mouths,
then split by video into train/val/test. Assumes 25 fps videos (as in NITYMED).
Output is NOT meant to be committed (NITYMED licence is CC BY-ND 4.0).
"""
import os, sys, glob, json, random, argparse
import pandas as pd, cv2
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_utils import process_video, HEAD_POSE_IDS
from labeling import label_frames
from extract_frames import video_meta

DEFAULT_ROOT = "/kaggle/input/datasets/nikospetrellis/nitymed/DSM_Dataset-HDTV720"


def process_one(path, step, feat_dir, crop_dir, seed):
    cls, sex, name = video_meta(path)
    base = name[:-4]
    rows, crops = [], {}
    for i, o in process_video(path, step=step):
        row = {"video": name, "cls": cls, "sex": sex, "frame": i, "face": o is not None}
        if o is not None:
            row["ear"], row["mar"] = o["ear"], o["mar"]
            for idx in HEAD_POSE_IDS:
                row[f"x{idx}"], row[f"y{idx}"] = o["pts"][idx]
            if o["mouth"] is not None:
                crops[i] = o["mouth"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{feat_dir}/{base}.csv", index=False)

    item = {"video": name, "n_seg": 0, "rows": []}
    if "mar" not in df.columns:                      # ما انلقى وجه أبداً
        return item
    lab, segs = label_frames(df, neg_ratio=1.0, rate=25 // step)
    item["n_seg"] = len(segs)
    if len(segs) != 3:                               # فيديو غامض: نستبعده
        return item

    by = dict(zip(df.frame.tolist(), lab.tolist()))
    pos = [f for f, l in by.items() if l == 1]
    neg = [f for f, l in by.items() if l == 0]
    random.Random(f"{seed}-{name}").shuffle(neg)     # ثابت لكل فيديو (يتحمل الاستكمال)
    keep = {f: 0 for f in neg[: 2 * len(pos)]}
    keep.update({f: 1 for f in pos})
    for f, l in sorted(keep.items()):
        if f not in crops:
            continue
        c = "yawn" if l else "no_yawn"
        p = f"{crop_dir}/{c}/{base}_{f}.png"
        cv2.imwrite(p, crops[f])
        item["rows"].append({"path": p, "video": name, "frame": f, "label": l})
    return item


def build(root, out, step=5, limit=None, seed=42):
    feat_dir, crop_dir = f"{out}/features", f"{out}/mouth"
    lab_dir, cache_dir = f"{out}/labels", f"{out}/cache"
    for d in (feat_dir, f"{crop_dir}/yawn", f"{crop_dir}/no_yawn", lab_dir, cache_dir):
        os.makedirs(d, exist_ok=True)

    vids = [v for v in glob.glob(root + "/**/*.mp4", recursive=True)
            if video_meta(v)[0] == "Yawning"]
    vids = sorted(vids, key=os.path.basename)
    if limit:
        vids = vids[:limit]
    print(f"عدد فيديوهات Yawning: {len(vids)}", flush=True)

    items = []
    for k, v in enumerate(vids, 1):
        name = os.path.basename(v)
        cache = f"{cache_dir}/{name[:-4]}.json"
        if os.path.exists(cache):                    # خلص من قبل
            items.append(json.load(open(cache)))
            continue
        item = process_one(v, step, feat_dir, crop_dir, seed)
        json.dump(item, open(cache, "w"))
        items.append(item)
        print(f"[{k}/{len(vids)}] {name}: مقاطع={item['n_seg']} صور={len(item['rows'])}", flush=True)

    pd.DataFrame([{"video": i["video"], "n_seg": i["n_seg"], "n_crops": len(i["rows"])}
                  for i in items]).to_csv(f"{lab_dir}/videos_summary.csv", index=False)

    used = [i for i in items if i["n_seg"] == 3]
    print(f"\nفيديوهات مستخدمة (3 مقاطع): {len(used)} من {len(items)}")
    meta = pd.DataFrame([r for i in used for r in i["rows"]])
    if len(used) < 3 or meta.empty:
        print("عدد الفيديوهات ما بكفي للتقسيم، وقفنا قبل التقسيم.")
        return
    meta.to_csv(f"{lab_dir}/mouth_all.csv", index=False)

    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, tmp = next(g1.split(meta, groups=meta.video))
    train, temp = meta.iloc[tr], meta.iloc[tmp].reset_index(drop=True)
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(g2.split(temp, groups=temp.video))
    val, test = temp.iloc[va], temp.iloc[te]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{lab_dir}/mouth_{name}.csv", index=False)
        print(f"{name:5s} | فيديوهات: {part.video.nunique():3d} | صور: {len(part):5d} | {part.label.value_counts().to_dict()}")
    assert not set(train.video) & set(val.video)
    assert not set(train.video) & set(test.video)
    assert not set(val.video) & set(test.video)
    print("تمام: ما في فيديو مكرر بين الأقسام\nالمخرجات في:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="/kaggle/working/nitymed_out")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="لتجربة سريعة على أول N فيديو")
    a = ap.parse_args()
    build(a.root, a.out, a.step, a.limit)
