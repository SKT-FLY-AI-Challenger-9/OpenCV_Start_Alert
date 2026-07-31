"""
OpenCV_Start_Alert 프로젝트 전역 설정.

아래 LIGHT 섹션은 R3가 사용한다. 검출 코드를 건드리지 않고도 R5/R6가
영상별로 임계값을 튜닝할 수 있도록 값을 이 파일에 모아뒀다.
"""

CONFIG = {
    "default_profile": "day",
    "profiles": {
        "day": {
            "light": {
                # (x, y, w, h), 전체 프레임 대비 비율(0~1).
                # 최종 영상이 확정되면 신호등 위치에 맞게 조정할 것.
                "roi": (0.35, 0.00, 0.30, 0.45),
                "min_area": 35,
                "min_ratio": 0.0015,
                "winner_margin": 1.25,
                "kernel_size": 5,
                "blur_size": 5,
                "red_ranges": [
                    ((0, 80, 80), (10, 255, 255)),
                    ((170, 80, 80), (179, 255, 255)),
                ],
                "green_ranges": [
                    ((35, 60, 70), (90, 255, 255)),
                ],
            }
        },
        "night": {
            "light": {
                "roi": (0.30, 0.00, 0.40, 0.50),
                "min_area": 20,
                "min_ratio": 0.0008,
                "winner_margin": 1.15,
                "kernel_size": 3,
                "blur_size": 3,
                "red_ranges": [
                    ((0, 60, 120), (12, 255, 255)),
                    ((168, 60, 120), (179, 255, 255)),
                ],
                "green_ranges": [
                    ((35, 45, 100), (95, 255, 255)),
                ],
            }
        },
    },
}
