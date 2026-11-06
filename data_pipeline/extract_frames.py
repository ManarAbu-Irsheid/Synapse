import os, glob, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_utils import process_video, HEAD_POSE_IDS

def video_meta(path):
    p = path.split("/")
    i = p.index("DSM_Dataset-HDTV720")
    return p[i + 1], p[i + 2], os.path.basename(path)      # class, sex, file

def extract_features(root, out_dir, step=5, only=None, classes=None):
    """ملف CSV لكل فيديو. only = أسماء ملفات، classes = مثلاً {'Yawning'}."""
    os.makedirs(out_dir, exist_ok=True)
    vids = sorted(glob.glob(root + "/**/*.mp4", recursive=True))
    if only:
        vids = [v for v in vids if os.path.basename(v) in only]
    if classes:
        vids = [v for v in vids if video_meta(v)[0] in classes]
    for k, v in enumerate(vids):
        cls, sex, name = video_meta(v)
        out_csv = os.path.join(out_dir, name.replace(".mp4", ".csv"))
        if os.path.exists(out_csv):
            continue                                        # مكتمل من قبل
        rows = []
        for i, out in process_video(v, step=step):
            row = {"video": name, "cls": cls, "sex": sex, "frame": i,
                   "face": out is not None}
            if out is not None:
                row["ear"], row["mar"] = out["ear"], out["mar"]
                pts = out["pts"]
                for idx in HEAD_POSE_IDS:
                    row[f"x{idx}"], row[f"y{idx}"] = pts[idx]
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{k+1}/{len(vids)}] {name}  ({len(rows)} rows)", flush=True)
