"""
Project-wide configuration for OpenCV_Start_Alert.

R3 uses the LIGHT section below. Values are intentionally grouped here so R5/R6
can tune thresholds per video without changing detection code.
"""

CONFIG = {
    "default_profile": "day",
    "profiles": {
        "day": {
            "light": {
                # (x, y, w, h) as ratios of the full frame.
                # Adjust this to the traffic signal area once final videos are fixed.
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
