import os, urllib.request
import cv2, numpy as np
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_Options = mp.tasks.vision.FaceLandmarkerOptions
_Mode = mp.tasks.vision.RunningMode

landmarker = _FaceLandmarker.create_from_options(_Options(
    base_options=_BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=_Mode.IMAGE,
    num_faces=1))

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 13, 14, 78, 308, 81, 178, 311, 402]
HEAD_POSE_IDS = [1, 152, 33, 263, 61, 291]   # أنف، ذقن، زوايا العينين، زوايا الفم

def get_landmarks(frame_bgr):
    """يرجع مصفوفة (N, 2) بالبكسل، أو None لو ما في وجه."""
    h, w = frame_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([(p.x * w, p.y * h) for p in lm])

def crop_roi(frame, pts, idxs, size, margin=0.4):
    """يقص مربع حول النقاط مع هامش ويعمل resize."""
    xy = pts[idxs]
    x1, y1 = xy.min(axis=0); x2, y2 = xy.max(axis=0)
    s = max(x2 - x1, y2 - y1) * (1 + margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = frame.shape[:2]
    a, b = max(0, int(cy - s/2)), min(h, int(cy + s/2))
    c, d = max(0, int(cx - s/2)), min(w, int(cx + s/2))
    roi = frame[a:b, c:d]
    return None if roi.size == 0 else cv2.resize(roi, size)

def preprocess_eye(roi_bgr, size=64):
    """نفس المعالجة بالتدريب والـ real-time: رمادي + resize + تطبيع."""
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size)).astype("float32") / 255.0
    return g[..., None]            # (64, 64, 1)

def preprocess_mouth(roi_bgr, size=96):
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)   # نفس ترتيب ألوان الـ loader
    return cv2.resize(rgb, (size, size)).astype("float32") / 255.0

def _d(a, b): return np.linalg.norm(a - b)

def eye_aspect_ratio(pts, idx):
    p = pts[idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]))

def mouth_aspect_ratio(pts):
    return _d(pts[13], pts[14]) / _d(pts[61], pts[291])

def extract_all(frame_bgr):
    """كل شي من فريم واحد: ROIs + EAR + MAR + النقاط."""
    pts = get_landmarks(frame_bgr)
    if pts is None:
        return None
    return {
        "left_eye":  crop_roi(frame_bgr, pts, LEFT_EYE,  (64, 64)),
        "right_eye": crop_roi(frame_bgr, pts, RIGHT_EYE, (64, 64)),
        "mouth":     crop_roi(frame_bgr, pts, MOUTH,     (96, 96)),
        "ear": (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2,
        "mar": mouth_aspect_ratio(pts),
        "pts": pts,
    }


# ================= Video / Real-time mode =================
def _features(frame_bgr, pts):
    """نفس مخرجات extract_all لكن من نقاط جاهزة."""
    return {
        "left_eye":  crop_roi(frame_bgr, pts, LEFT_EYE,  (64, 64)),
        "right_eye": crop_roi(frame_bgr, pts, RIGHT_EYE, (64, 64)),
        "mouth":     crop_roi(frame_bgr, pts, MOUTH,     (96, 96)),
        "ear": (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2,
        "mar": mouth_aspect_ratio(pts),
        "pts": pts,
    }

class FaceTracker:
    """MediaPipe بوضع VIDEO: لفريمات متتالية (فيديو أو كاميرا)."""
    def __init__(self, num_faces=1):
        self.lm = _FaceLandmarker.create_from_options(_Options(
            base_options=_BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=_Mode.VIDEO,
            num_faces=num_faces))
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms):
        """يرجع نفس dict تبع extract_all، أو None لو ما في وجه."""
        ts = int(timestamp_ms)
        if ts <= self._last_ts:          # لازم يزيد دايماً
            ts = self._last_ts + 1
        self._last_ts = ts
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            return None
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
        return _features(frame_bgr, pts)

    def close(self):
        self.lm.close()

def process_video(path, step=1):
    """يمر على فيديو ويرجع (رقم الفريم، النتيجة). step=5 يعني كل 5 فريمات."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = FaceTracker()
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i, tracker.process(frame, i / fps * 1000)
            i += 1
    finally:
        cap.release()
        tracker.close()
