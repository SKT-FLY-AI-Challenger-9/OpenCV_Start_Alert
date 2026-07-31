# -*- coding: utf-8 -*-
"""
선택 심화 — YOLO 객체 탐지 (Vehicle / Light Detection by YOLO)
==============================================================
교안 16p '도전 과제 · YOLO 차량 검출 — 앞차를 자동 인식해 ROI를 스스로 설정'.

이 모듈은 검출 주력이 아니다. 미션 A/B 의 판정은 그대로 R2(absdiff) · R3(HSV)가
담당하고, 여기서는 **ROI 를 자동으로 잡아주는 일**만 한다.
따라서 이 파일을 통째로 빼도 파이프라인은 수동 ROI 로 그대로 돈다.

공개 API
    get_vehicle_roi(frame, prev=None) -> (x, y, w, h)   0~1 정규화
    get_light_roi(frame)              -> (x, y, w, h) | None
    detect(frame)                     -> [{"name","conf","box"}]
    draw_detections(frame, ...)       -> 오버레이

ROI 반환 형식은 config.py / src/light.py `_crop_roi()` / src/motion.py
`_roi_to_pixels()` 와 동일한 **정규화 (x, y, w, h)** 다. 좌상단 시작 비율 +
폭·높이 비율이며, 두 꼭짓점이 아니다. 해상도가 달라도 그대로 쓸 수 있다.

ultralytics 가 없거나 모델 로드에 실패해도 import·호출이 죽지 않는다.
그 경우 detect() 는 빈 리스트를, get_vehicle_roi() 는 프로파일의 폴백 ROI 를 준다.
"""

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================== CONFIG ==============================
USE_YOLO = True
MODEL_PATH = "yolo26n.pt"
MODEL_FALLBACKS = ["yolo11n.pt", "yolov8n.pt"]

IMG_SIZE = 640
DEVICE = "cpu"              # GPU 있으면 0

# 락온 주기. 정차 상황이라 앞차 위치가 고정이므로 매 프레임 돌릴 이유가 없다.
# 매 프레임 추론하면 박스가 미세하게 흔들려 absdiff 화소 수에 노이즈가 섞인다.
YOLO_EVERY_N = 30

ROI_MARGIN = 0.10           # 앞차 박스를 이 비율만큼 넓혀 ROI 로 쓴다
MAX_BOX_W_RATIO = 0.75      # 프레임 폭의 이 비율을 넘는 박스는 오탐 (실측: w=1920 오탐 존재)
MIN_BOX_H_RATIO = 0.18      # 실측 - 진짜 앞차 31~35%, 원거리 차 11~12% 로 확실히 갈린다

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
LIGHT_CLASSES = {"traffic light"}


# ------------------------- 영상별 프로파일 -------------------------
# FRONT_BAND = (좌, 우, 상, 하) 정규화. 이 안에 중심이 있는 차량만 앞차 후보로 본다.
#              옆 차선 차량을 걸러내는 장치. 아래 값은 전부 실측 기준이다.
# FALLBACK   = YOLO 실패 시 쓸 정규화 ROI. 정차 구간 앞차 박스를 10% 확장한 값.
PROFILES = {
    # 영상 1 Day_Car — 앞차 center=(0.48,0.69). 옆 차선 차가 0.18/0.77/0.93 에 잡혀
    # 우측 경계를 0.72 로 조였다.
    "day": {
        "conf": 0.35,
        "front_band": (0.30, 0.72, 0.45, 0.95),
        "fallback": (0.252, 0.499, 0.466, 0.376),
        "light_conf": 0.20,
        "light_roi": None,
    },
    # 영상 2 Night_Car — 앞차 center=(0.52,0.54).
    # 좌측 차선 흰 세단이 앞차보다 '더 크게'(면적 9.5% vs 7.8%) 잡힌다.
    # 면적만 보면 옆차를 앞차로 착각하므로 좌측 경계가 반드시 필요하다.
    "night": {
        "conf": 0.35,
        "front_band": (0.35, 0.75, 0.35, 0.90),
        "fallback": (0.385, 0.331, 0.269, 0.416),
        "light_conf": 0.20,
        "light_roi": None,
    },
    # 영상 3 Night_Nocar — 앞차 없음. 신호등 3등화가 화면 상단에 있다.
    # YOLO 의 traffic light 는 conf 0.20~0.43 으로 낮고 프레임마다 위치가 튀어서
    # (40프레임 중 최다 셀도 31회) 위치는 고정 ROI 로 두는 편이 안정적이다.
    "night_nocar": {
        "conf": 0.30,
        "front_band": (0.35, 0.75, 0.35, 0.90),
        "fallback": (0.396, 0.398, 0.208, 0.278),
        "light_conf": 0.18,
        "light_roi": (0.396, 0.093, 0.250, 0.352),
    },
}

_active = "day"
_P = PROFILES[_active]
# ====================================================================


_model = None
_model_failed = False
_frame_idx = 0
_locked = None
_last_dets = []


def set_profile(name):
    """영상별 파라미터를 전환한다. 'day' / 'night' / 'night_nocar'"""
    global _active, _P
    if name not in PROFILES:
        raise ValueError(f"모르는 프로파일: {name} (가능: {list(PROFILES)})")
    _active, _P = name, PROFILES[name]
    reset()


def reset():
    """영상을 바꿔 실행할 때 락온 상태를 초기화한다."""
    global _frame_idx, _locked, _last_dets
    _frame_idx, _locked, _last_dets = 0, None, []


def available():
    """YOLO 를 실제로 쓸 수 있는 상태인지."""
    return USE_YOLO and _load_model() is not None


def _load_model():
    """지연 로드. 실패하면 None 을 돌려주고 다시 시도하지 않는다."""
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    if YOLO is None:
        print("[yolo] ultralytics 미설치 -> 수동 ROI 로 폴백")
        _model_failed = True
        return None
    for path in [MODEL_PATH] + MODEL_FALLBACKS:
        try:
            _model = YOLO(path)
            print(f"[yolo] 모델 로드: {path}")
            return _model
        except Exception as e:
            print(f"[yolo] 로드 실패 ({path}): {e}")
    print("[yolo] 전체 로드 실패 -> 수동 ROI 로 폴백")
    _model_failed = True
    return None


def detect(frame, conf=None):
    """프레임에서 객체를 탐지한다.

    Returns
    -------
    list[dict]
        [{"name": str, "conf": float, "box": (x, y, w, h)}] — box 는 픽셀 좌표.
        탐지 불가 상황에서는 빈 리스트.
    """
    if not USE_YOLO or frame is None:
        return []
    model = _load_model()
    if model is None:
        return []
    try:
        results = model.predict(frame, conf=_P["conf"] if conf is None else conf,
                                imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    except Exception as e:
        print(f"[yolo] 추론 실패: {e}")
        return []

    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({"name": r.names[int(b.cls[0])],
                         "conf": float(b.conf[0]),
                         "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1))})
    return dets


def _pick_front(dets, shape):
    """front_band 안에서 면적이 가장 큰 차량을 앞차로 고른다.
    (가까울수록 크게 잡히므로 면적이 거리 대용이다)"""
    h, w = shape[:2]
    lx, rx, ty, by = _P["front_band"]
    best, best_area = None, 0
    for d in dets:
        if d["name"] not in VEHICLE_CLASSES:
            continue
        x, y, bw, bh = d["box"]
        if bw > w * MAX_BOX_W_RATIO:      # 화면을 덮는 오탐
            continue
        if bh < h * MIN_BOX_H_RATIO:      # 멀리 있는 차는 앞차가 아니다
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        if not (lx <= cx <= rx and ty <= cy <= by):
            continue
        if bw * bh > best_area:
            best, best_area = d, bw * bh
    return best


def _to_norm(box, shape, margin=ROI_MARGIN):
    """픽셀 박스를 margin 만큼 넓혀 정규화 (x, y, w, h) 로 바꾼다."""
    h, w = shape[:2]
    x, y, bw, bh = box
    dx, dy = bw * margin, bh * margin
    x0, y0 = max(0.0, x - dx), max(0.0, y - dy)
    x1, y1 = min(float(w), x + bw + dx), min(float(h), y + bh + dy)
    return (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def get_vehicle_roi(frame, prev=None):
    """앞차 ROI 를 정규화 (x, y, w, h) 로 돌려준다.

    config.py 의 ROI 와 형식이 같으므로 main 에서 한 줄만 바꾸면 된다.

        roi = vehicle_yolo.get_vehicle_roi(color) if USE_YOLO else CFG["roi"]

    탐지에 실패하면 이전 ROI -> 프로파일 폴백 순으로 내려간다. 절대 None 이 아니다.
    """
    global _frame_idx, _locked, _last_dets

    fallback = prev or _locked or _P["fallback"]
    if not USE_YOLO:
        return _P["fallback"]

    # 락온 주기는 탐지 성공 여부와 무관하게 지킨다.
    # 실패를 조건에 넣으면 앞차 없는 영상에서 매 프레임 추론이 돌아 3배 느려진다.
    period = YOLO_EVERY_N if YOLO_EVERY_N > 0 else 10 ** 9
    need = (_frame_idx % period == 0)
    _frame_idx += 1
    if not need:
        return fallback

    dets = detect(frame)
    if dets:
        _last_dets = dets
    front = _pick_front(dets, frame.shape)
    if front is None:
        return fallback

    _locked = _to_norm(front["box"], frame.shape)
    return _locked


def get_light_roi(frame):
    """신호등 ROI 를 정규화 (x, y, w, h) 로 돌려준다. 없으면 None.

    프로파일에 light_roi 가 있으면 그 고정값을 우선 쓴다. YOLO 의 traffic light 는
    conf 가 낮고 위치가 튀어서 고정 ROI 쪽이 훨씬 안정적이다.
    """
    if _P["light_roi"] is not None:
        return _P["light_roi"]
    if not USE_YOLO:
        return None
    lights = [d for d in detect(frame, conf=_P["light_conf"])
              if d["name"] in LIGHT_CLASSES]
    if not lights:
        return None
    best = max(lights, key=lambda d: d["conf"])
    return _to_norm(best["box"], frame.shape, margin=0.20)


def last_detections():
    """최근 탐지 결과. 매 프레임 그릴 때 재추론 없이 쓴다."""
    return _last_dets


def draw_detections(frame, dets=None):
    """탐지 박스를 그린다 (원본을 직접 수정)."""
    for d in (last_detections() if dets is None else dets):
        x, y, w, h = d["box"]
        if d["name"] in VEHICLE_CLASSES:
            color = (0, 200, 255)
        elif d["name"] in LIGHT_CLASSES:
            color = (0, 255, 120)
        else:
            color = (160, 160, 160)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{d['name']} {d['conf']:.2f}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame