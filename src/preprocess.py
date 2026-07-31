"""
R1 - 전처리 담당 (Preprocessing)
출발 알림 시스템 - preprocess(frame) 구현

역할: 원본 컬러 프레임을 받아 그레이+블러 처리된 프레임을 반환.
      해상도 통일, 낮/밤 자동 판별, 밤 영상 밝기 보정까지 포함.
"""

import cv2
import numpy as np

# ------------------------------------------------------------------
# CONFIG: 팀 전체가 참고할 설정값. 임계값·ROI는 전부 여기 모아둔다.
# ------------------------------------------------------------------
CONFIG = {
    "target_size": (1280, 720),   # 영상마다 해상도가 달라 표준 크기로 통일
    "brightness_threshold": 80,   # 이 값보다 어두우면 '밤'으로 판단
    "day": {
        "blur_ksize": (5, 5),
    },
    "night": {
        "blur_ksize": (5, 5),
        "clahe_clip": 2.5,        # 밤 영상 대비 보정 강도
        "clahe_grid": (8, 8),
    },
}

# ROI는 target_size(1280x720) 기준 좌표. 실제 영상 확인 후 정한 값.
ROI = {
    "car":   (420, 300, 900, 650),   # 앞차 영역 (x1, y1, x2, y2)
    "light": (480, 0, 970, 320),     # 신호등 영역 (x1, y1, x2, y2)
}


def dummy_preprocess(frame):
    """
    더미 버전. 킥오프 직후 R2·R3가 바로 통합 테스트할 수 있도록
    최소한의 동작만 하는 껍데기.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def is_night(gray_frame, threshold=None):
    """평균 밝기로 낮/밤 자동 판별."""
    threshold = threshold or CONFIG["brightness_threshold"]
    return gray_frame.mean() < threshold


def preprocess(frame, mode=None):
    """
    R1 담당 핵심 함수.

    Parameters
    ----------
    frame : np.ndarray (BGR)
        원본 컬러 프레임
    mode : str or None
        'day' / 'night' 강제 지정. None이면 밝기로 자동 판별.

    Returns
    -------
    gray : np.ndarray (1채널)
        resize + grayscale + (밤이면 대비보정) + blur 처리된 프레임
    meta : dict
        - mode: 'day' / 'night' 판별 결과
        - resized_shape: 리사이즈된 프레임 크기
        - color: 리사이즈만 된 컬러 프레임 (YOLO 등 컬러 입력이 필요한 검출기용)
    """
    # 1) 해상도 통일 (영상마다 1280x720 / 1920x1080으로 달라서 필요)
    resized = cv2.resize(frame, CONFIG["target_size"], interpolation=cv2.INTER_AREA)

    # 2) 그레이 변환
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3) 낮/밤 판별
    if mode is None:
        mode = "night" if is_night(gray) else "day"

    # 4) 밤이면 CLAHE로 대비 보정 (단순 normalize보다 국소 대비에 강함)
    if mode == "night":
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG["night"]["clahe_clip"],
            tileGridSize=CONFIG["night"]["clahe_grid"],
        )
        gray = clahe.apply(gray)
        ksize = CONFIG["night"]["blur_ksize"]
    else:
        ksize = CONFIG["day"]["blur_ksize"]

    # 5) 블러로 노이즈 제거
    gray = cv2.GaussianBlur(gray, ksize, 0)

    # 6) 컬러 원본(리사이즈만 된 상태)도 함께 넘김
    #    -> R2·R3가 absdiff/HSV 방식을 쓰면 gray를 쓰면 되고,
    #       YOLO 등 컬러 입력이 필요한 검출기를 쓰면 meta["color"]를 꺼내 쓰면 됨.
    #       즉 흑백/블러 처리는 "선택적 재료"이지 강제가 아님.
    meta = {"mode": mode, "resized_shape": resized.shape, "color": resized}
    return gray, meta


def get_roi(gray_or_frame, region):
    """
    ROI 슬라이싱. region은 'car' 또는 'light'.
    preprocess()를 거친(=target_size로 resize된) 프레임에 적용해야 좌표가 맞는다.
    """
    x1, y1, x2, y2 = ROI[region]
    return gray_or_frame[y1:y2, x1:x2]


def draw_roi_debug(frame_bgr, region):
    """ROI 확인용 시각화 (디버깅 창에 띄우기)."""
    resized = cv2.resize(frame_bgr, CONFIG["target_size"], interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = ROI[region]
    vis = resized.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


if __name__ == "__main__":
    # 간단한 자체 테스트: frames/*.jpg 로 확인
    import os

    test_images = {
        "day_car_t5.jpg": "car",
        "night_car_t5.jpg": "car",
        "night_nocar_t5.jpg": "light",
    }

    os.makedirs("preprocess_test_out", exist_ok=True)

    for fname, region in test_images.items():
        path = os.path.join("frames", fname)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] {path} 없음, 건너뜀")
            continue

        gray, meta = preprocess(frame)
        roi_crop = get_roi(gray, region)
        roi_debug = draw_roi_debug(frame, region)

        base = fname.replace(".jpg", "")
        cv2.imwrite(f"preprocess_test_out/{base}_gray.jpg", gray)
        cv2.imwrite(f"preprocess_test_out/{base}_roi_crop.jpg", roi_crop)
        cv2.imwrite(f"preprocess_test_out/{base}_roi_debug.jpg", roi_debug)

        print(f"{fname}: mode={meta['mode']}, mean_brightness(after)={gray.mean():.1f}")
