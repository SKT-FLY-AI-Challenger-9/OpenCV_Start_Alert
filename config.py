# -*- coding: utf-8 -*-
"""프로젝트 전체 설정.

ROI는 (x1, y1, x2, y2)의 정규화 좌표임.
예: (0.2, 0.3, 0.8, 0.9)는 영상 너비/높이의 20~80%, 30~90% 영역을 의미함.

주의:
- 아래 ROI와 임계값은 실행 가능한 예시값임.
- 실제 제공 영상에서 반드시 다시 조정해야 함.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT_DIR / "videos"
OUTPUT_DIR = ROOT_DIR / "outputs"

PROFILES: dict[str, dict[str, Any]] = {
    "day_car": {
        "label": "영상 1 · 낮 · 앞차 있음",
        "mission": "A",
        "video_path": VIDEO_DIR / "Day_Car.mp4",
        "output_path": OUTPUT_DIR / "Day_Car_result.mp4",
        "preprocess": {
            "roi": (0.20, 0.22, 0.80, 0.90),
            "resize_width": 640,
            "brightness_alpha": 1.0,
            "brightness_beta": 0,
            "blur_kernel": 5,
        },
        "motion": {
            "diff_threshold": 25,
            "morph_kernel": 5,
            "open_iterations": 1,
            "dilate_iterations": 2,
            "min_motion_ratio": 0.025,
        },
        "decision": {
            "arming_still_frames": 10,
            "required_frames": 4,
        },
    },
    "night_car": {
        "label": "영상 2 · 밤 · 앞차 있음",
        "mission": "A",
        # 제공 파일명이 Nigt_Car.mp4로 되어 있어 그대로 사용함.
        "video_path": VIDEO_DIR / "Nigt_Car.mp4",
        "output_path": OUTPUT_DIR / "Nigt_Car_result.mp4",
        "preprocess": {
            "roi": (0.20, 0.22, 0.80, 0.90),
            "resize_width": 640,
            "brightness_alpha": 1.25,
            "brightness_beta": 12,
            "blur_kernel": 5,
        },
        "motion": {
            "diff_threshold": 18,
            "morph_kernel": 5,
            "open_iterations": 1,
            "dilate_iterations": 2,
            "min_motion_ratio": 0.018,
        },
        "decision": {
            "arming_still_frames": 12,
            "required_frames": 5,
        },
    },
    "night_nocar": {
        "label": "영상 3 · 밤 · 앞차 없음",
        "mission": "B",
        "video_path": VIDEO_DIR / "Night_Nocar.mp4",
        "output_path": OUTPUT_DIR / "Night_Nocar_result.mp4",
        "preprocess": {
            # 신호등이 위치한 화면 상단 영역의 예시값
            "roi": (0.50, 0.02, 0.98, 0.55),
            "resize_width": 500,
            "brightness_alpha": 1.0,
            "brightness_beta": 0,
            "blur_kernel": 5,
        },
        "light": {
            "red_ranges": [
                ((0, 80, 80), (10, 255, 255)),
                ((170, 80, 80), (179, 255, 255)),
            ],
            "green_range": ((35, 70, 70), (90, 255, 255)),
            "morph_kernel": 5,
            "open_iterations": 1,
            "close_iterations": 1,
            "min_color_ratio": 0.0020,
            "dominance_ratio": 1.20,
            "use_hough": False,
            "hough": {
                "dp": 1.0,
                "min_dist": 20,
                "param1": 100,
                "param2": 12,
                "min_radius": 3,
                "max_radius": 30,
            },
        },
        "decision": {
            "red_required_frames": 3,
            "green_required_frames": 4,
        },
    },
}


def get_profile(name: str) -> dict[str, Any]:
    """설정 원본이 수정되지 않도록 복사본을 반환함."""
    if name not in PROFILES:
        raise KeyError(f"알 수 없는 프로필: {name}")
    return deepcopy(PROFILES[name])
