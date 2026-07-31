"""
R4 — 판정 · 알림 담당 (Decision & Alert)

R2(motion.py)와 R3(light.py)가 넘겨주는 evidence를 보고
'출발' 여부를 확정(N프레임 연속 조건)하고, 화면에 알림을 그린다.

evidence 계약 (R2/R3와 합의한 최소 규격):
    motion_pixels : int        - ROI 안 움직인 화소 수 (미션 A, R2의 detect_motion() 반환값)
    light_color   : str | None - 'RED' / 'GREEN' / None (미션 B, R3가 채움)
    side_clear    : bool       - 주변 차량 정지 여부, 기본 True (미션 B, R3, 선택)

R2/R3가 아직 더미 상태여도 위 키만 지키면 이 파일은 그대로 동작한다.
"""

import os
import time
import cv2

STOPPED = "STOPPED"
CANDIDATE = "CANDIDATE"
STARTED = "STARTED"

# ------------------------------------------------------------
# CONFIG: 낮/밤 파라미터 분리 (R2의 MOTION_CONFIG와 동일한 방식)
# R2는 diff_thresh(이진화 임계값)를 day/night로 나누고, R4는 그 결과로 나온
# '움직인 화소 수'가 얼마나 쌓여야 출발로 확정할지를 day/night로 나눈다.
# 밤은 차체가 어둡고 후미등만 반응하므로 R2가 넘기는 pixels 자체가 낮게 나온다
# (교안 7p "밤 영상은 후미등 중심으로 재튜닝" 대응) -> 여기 임계값도 같이 낮춰야 함.
# night 값은 실제 영상 2로 확인 전까지는 플레이스홀더이며 1차 통합 시 재튜닝 대상.
# ------------------------------------------------------------
DECISION_CONFIG = {
    "day": {
        "motion_pixel_threshold": 2000,
        "consecutive_frames_required": 4,
    },
    "night": {
        "motion_pixel_threshold": 400,
        "consecutive_frames_required": 4,
    },
}

class DecisionState:
    """미션별 판정 상태. main 루프에서 프레임마다 decide()에 넘긴다.

    mode는 R2의 detect_motion(mode=...)와 동일한 값("day"/"night")을 넘기면
    DECISION_CONFIG에서 해당 임계값을 자동으로 가져온다.
    """

    def __init__(self, mode: str = "day"):
        cfg = DECISION_CONFIG[mode]
        self.mode = mode
        self.consecutive_frames_required = cfg["consecutive_frames_required"]
        self.motion_pixel_threshold = cfg["motion_pixel_threshold"]
        self.status = STOPPED
        self.consecutive_count = 0
        self.fired = False  # 이번 정차 사이클에서 이미 알림을 띄웠는지 (중복 알림 방지)

def decide(state: DecisionState, evidence: dict) -> bool:
    """
    누적 상태(state)와 이번 프레임의 evidence를 보고 '출발' 알림을 새로 확정하면 True.

    - light_color가 주어지면 미션 B(신호등) 판정, 없으면 미션 A(움직임) 판정.
    - candidate 조건이 consecutive_frames_required 프레임 연속으로 유지되어야 확정.
    - 한 번 확정된 뒤에는 reset()을 호출하기 전까지 다시 True를 반환하지 않는다.
    """
    motion_pixels = evidence.get("motion_pixels", 0)
    light_color = evidence.get("light_color")
    side_clear = evidence.get("side_clear", True)

    if light_color is not None:
        candidate = light_color == "GREEN" and side_clear
    else:
        candidate = motion_pixels > state.motion_pixel_threshold

    if candidate:
        state.consecutive_count += 1
        state.status = CANDIDATE
    else:
        state.consecutive_count = 0
        state.status = STOPPED

    if state.consecutive_count >= state.consecutive_frames_required:
        state.status = STARTED
        if not state.fired:
            state.fired = True
            return True

    return False

def reset(state: DecisionState):
    """다음 정차 사이클을 위해 상태를 초기화한다 (알림 후 다시 정지 상태로 돌아왔을 때 호출)."""
    state.status = STOPPED
    state.consecutive_count = 0
    state.fired = False

def draw_alert(frame, fired: bool, alert_config: dict, save: bool = False, output_dir: str = "outputs"):
    """
    fired가 True인 프레임에서 호출. 화면 오버레이 + 로그 출력, 선택적으로 이미지 저장.
    fired가 False면 프레임을 그대로 반환한다 (알림 없음).
    """
    if not fired:
        return frame

    text = alert_config.get("text", "GO!")
    font_scale = alert_config.get("font_scale", 1.2)
    color = alert_config.get("color_bgr", (0, 255, 0))
    thickness = alert_config.get("thickness", 3)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    print(f"[ALERT] {text} @ {time.strftime('%H:%M:%S')}")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"alert_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(filename, frame)

    return frame

def _run_mode_demo(mode: str, pixels_stopped: int, pixels_moving: int):
    """R2의 detect_motion() 없이도 R4 로직만 확인하는 모드별 더미 데모.
    정지 -> 순간 튐(오탐 무시) -> 진짜 출발(연속 유지) 순서로 evidence를 흉내낸다."""
    dummy_evidence_stream = (
        [{"motion_pixels": pixels_stopped}] * 5
        + [{"motion_pixels": pixels_moving}] * 1   # 노이즈 1프레임 (오탐이면 안 됨)
        + [{"motion_pixels": pixels_stopped}] * 2  # 다시 정지
        + [{"motion_pixels": pixels_moving}] * 5   # 실제 출발 (연속 유지)
    )

    state = DecisionState(mode=mode)
    print(f"\n--- mode={mode} (threshold={state.motion_pixel_threshold}) ---")
    for i, evidence in enumerate(dummy_evidence_stream):
        fired = decide(state, evidence)
        print(f"frame={i:02d} evidence={evidence} status={state.status} fired={fired}")


if __name__ == "__main__":
    # day/night 각각 DECISION_CONFIG 임계값이 R2의 화소 수 규모와 맞는지 확인.
    _run_mode_demo(mode="day", pixels_stopped=50, pixels_moving=3000)
    _run_mode_demo(mode="night", pixels_stopped=20, pixels_moving=600)
