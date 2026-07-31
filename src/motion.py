# -*- coding: utf-8 -*-
"""
R2 - 움직임 검출 담당 (Motion Detection)
========================================
팀 인터페이스 계약 : pixels = detect_motion(prev, cur)

담당 범위
  - absdiff 차영상 구현
  - threshold / 모폴로지 정리
  - 움직임 화소 수 계산
  - 미션 A(앞차 출발 알림) 검출 전담

활용 강의 : [5] 회선 처리(모폴로지) · [5-2] 차영상 미니프로젝트 · [9] 앞차량 움직임 감지

보조 함수
  - detect_taillight_mask() : 밤 영상(영상 2)에서 후미등(빨강)을 HSV inRange로 잡아
    움직임 검출 ROI를 좁히는 용도. (교안 7p "밤 영상은 후미등 중심으로 재튜닝" 대응)
"""
import cv2
import numpy as np

# ------------------------------------------------------------
# CONFIG : 낮/밤 파라미터 분리 (요구사항 필수-7)
# 다른 팀원(R4/R5)이 통합 CONFIG로 옮길 수 있도록 값만 이 블록에 모아둔다.
# ------------------------------------------------------------
MOTION_CONFIG = {
    "day": {
        "blur_ksize": 5,
        "diff_thresh": 25,     # 낮: 대비가 커서 임계값을 높게 잡아도 검출됨
        "morph_ksize": 5,
        "morph_iter": 2,
        "min_area": 60,        # 이 면적보다 작은 컨투어는 잡음으로 제거
    },
    "night": {
        "blur_ksize": 5,
        "diff_thresh": 12,     # 밤: 차체가 어두워 반응이 약함 -> 임계값을 낮춤
        "morph_ksize": 3,
        "morph_iter": 1,
        "min_area": 25,
    },
}

# 후미등(빨강) HSV 범위. Hue가 0도를 걸치므로 두 구간을 bitwise_or로 합친다.
TAILLIGHT_HSV = {
    "lower_red1": (0, 90, 90),
    "upper_red1": (10, 255, 255),
    "lower_red2": (170, 90, 90),
    "upper_red2": (179, 255, 255),
}


def _to_gray_blur(frame, ksize):
    """컬러 프레임이 들어오면 그레이 변환 + 블러까지 처리한다.
    (R1의 preprocess()가 아직 없을 때도 이 모듈 단독으로 테스트 가능하게 하기 위함)"""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ksize > 0:
        frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return frame


def _remove_small_blobs(mask, min_area):
    """contourArea 기준으로 작은 잡음 덩어리를 지운다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean


def _roi_to_pixels(frame_shape, roi):
    """정규화 ROI (x, y, w, h), 0~1 -> 절대 픽셀 슬라이스 좌표 (x1, y1, x2, y2).

    (x, y)는 좌상단 시작 비율, (w, h)는 폭/높이 비율이다 (두 꼭짓점이 아님).
    config.py CONFIG['profiles'][mode]['light']['roi']와 src/light.py의 _crop_roi()가
    쓰는 표기와 동일한 규약으로 맞췄다. 해상도가 달라져도(day/night_car=640,
    night_nocar=500) 같은 비율을 그대로 재사용할 수 있다.
    """
    h, w = frame_shape[:2]
    x, y, rw, rh = roi
    px1 = max(0, min(w, int(round(x * w))))
    py1 = max(0, min(h, int(round(y * h))))
    px2 = max(0, min(w, int(round((x + rw) * w))))
    py2 = max(0, min(h, int(round((y + rh) * h))))
    return px1, py1, px2, py2


def detect_motion(prev, cur, mode="day", roi=None, return_mask=False):
    """이전 프레임(prev)과 현재 프레임(cur)의 차영상으로 움직임을 검출한다.

    절차 : absdiff -> threshold -> morphologyEx(OPEN/CLOSE) -> 작은 잡음 제거 -> 화소 수

    Parameters
    ----------
    prev, cur : np.ndarray
        연속한 두 프레임. BGR 컬러 / 그레이 모두 허용.
    mode : "day" | "night"
        MOTION_CONFIG 에서 사용할 파라미터셋.
    roi : (x, y, w, h) | None
        0~1로 정규화된 좌표 (좌상단 시작 비율 + 폭/높이 비율). config.py 및
        src/light.py의 _crop_roi()와 동일한 형식이며, 지정하면 해당 영역만
        잘라 분석한다 (연산량 절감 + 오탐 감소).
    return_mask : bool
        True면 (pixels, mask) 튜플을, False면 pixels(int)만 반환.
        ※ 팀 인터페이스 계약은 `pixels = detect_motion(prev, cur)` 이므로
          다른 모듈에서 호출할 때는 기본값(False)을 그대로 쓰면 된다.

    Returns
    -------
    int 또는 (int, np.ndarray)
        움직임으로 판정된 흰색 화소 수 (+ 정리된 이진 마스크)
    """
    cfg = MOTION_CONFIG[mode]

    if roi is not None:
        x1, y1, x2, y2 = _roi_to_pixels(prev.shape, roi)
        prev = prev[y1:y2, x1:x2]
        cur = cur[y1:y2, x1:x2]

    gray_prev = _to_gray_blur(prev, cfg["blur_ksize"])
    gray_cur = _to_gray_blur(cur, cfg["blur_ksize"])

    diff = cv2.absdiff(gray_prev, gray_cur)
    _, mask = cv2.threshold(diff, cfg["diff_thresh"], 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_ksize"], cfg["morph_ksize"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg["morph_iter"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg["morph_iter"])

    mask = _remove_small_blobs(mask, cfg["min_area"])

    pixels = int(cv2.countNonZero(mask))

    if return_mask:
        return pixels, mask
    return pixels


def detect_taillight_mask(frame, roi=None):
    """밤 영상(영상 2)에서 후미등(빨강)을 HSV inRange로 검출한다.

    detect_motion()의 roi를 이 마스크의 바운딩 영역으로 좁혀서 넘기면
    밤 영상에서 차체가 아닌 후미등 주변 움직임에 집중할 수 있다.

    Parameters
    ----------
    frame : np.ndarray
        BGR 컬러 프레임.
    roi : (x, y, w, h) | None
        0~1로 정규화된 좌표 (좌상단 시작 비율 + 폭/높이 비율, config.py와 동일 형식).
        지정하면 해당 영역만 잘라 분석.

    Returns
    -------
    np.ndarray
        후미등(빨강) 영역이 흰색(255)인 이진 마스크.
    """
    if roi is not None:
        x1, y1, x2, y2 = _roi_to_pixels(frame.shape, roi)
        frame = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array(TAILLIGHT_HSV["lower_red1"]),
        np.array(TAILLIGHT_HSV["upper_red1"]),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array(TAILLIGHT_HSV["lower_red2"]),
        np.array(TAILLIGHT_HSV["upper_red2"]),
    )
    return cv2.bitwise_or(mask1, mask2)
