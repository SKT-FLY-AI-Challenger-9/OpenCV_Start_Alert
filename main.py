# -*- coding: utf-8 -*-
"""
R5 - 통합 · 발표 담당 (main.py)

R1(preprocess) ~ R4(decision) 를 엮어 영상 3종을 끝까지 돌리는 진입점.
선택 심화인 R-YOLO(src/vehicle_yolo.py) 는 ROI 자동 설정에만 관여하며,
config.USE_YOLO = False 로 두면 수동 ROI 로 그대로 동작한다.

파이프라인
    frame
      -> R1 preprocess()            해상도 통일(1280x720) · 그레이 · 밤이면 CLAHE · 블러
      -> YOLO get_vehicle_roi()     앞차 ROI 자동 설정 (실패 시 config 의 수동 ROI)
      -> R2 detect_motion()         absdiff · threshold · 모폴로지 · 화소 수      [미션 A]
         R3 detect_light()          HSV inRange · 등화 색 판정                    [미션 B]
      -> R4 decide()                임계값 + N프레임 연속 + 상태 전이
      -> R4 draw_alert()            오버레이 · 소리 · 저장

사용법
    python main.py                      # 3종 전부, 화면 없이 outputs/ 에 저장
    python main.py day                  # 영상 1만
    python main.py night --show         # 화면 보면서 (q 로 종료)
    python main.py --no-save --show
    python main.py --probe              # 판정 없이 화소 수 통계만 출력 (튜닝용)
"""

import contextlib
import io
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.preprocess import preprocess
from src.motion import detect_motion, _roi_to_pixels
from src.light import detect_light
from src.decision import (
    DecisionState, decide, draw_alert, draw_perf_hud, STARTED,
)
from src import vehicle_yolo


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def find_video(name):
    for d in config.VIDEO_DIRS:
        p = d if os.path.isabs(d) else os.path.join(ROOT, d)
        p = os.path.join(p, name)
        if os.path.exists(p):
            return p
    return None


def roi_to_px(roi, shape):
    """정규화 (x, y, w, h) -> 픽셀 (x1, y1, x2, y2).

    R2 의 _roi_to_pixels 를 그대로 재사용한다. 자체 구현하면 반올림 방식이 달라져
    마스크와 슬라이스 크기가 1픽셀씩 어긋난다 (실제로 IndexError 로 터졌다).
    """
    return _roi_to_pixels(shape, roi)


class Pacer:
    """--realtime 일 때 재생 속도를 원본 fps 에 맞춘다.

    imshow + waitKey(1) 만 쓰면 '처리되는 대로' 재생돼서 낮 영상(24fps)은
    2배 이상 빨리 지나간다. 시연에서는 원래 속도로 보여야 한다.

    반대로 밤 영상(1920x1080 @ 59.7fps)은 처리가 원본 속도를 못 따라간다.
    이때는 대기 없이 통과시키므로 원본보다 느리게 재생된다. 모든 프레임을 다
    보여주는 쪽이 낫다고 판단했다 — 밀린 프레임을 버려봤더니 창이 거의 정지해서
    시연에 훨씬 나빴다. (검출·판정 결과는 재생 속도와 무관하게 동일하다)
    """

    def __init__(self, fps, enabled):
        self.dt = 1.0 / fps if fps else 0
        self.enabled = enabled
        self.t0 = None
        self.shown = 0
        self.dropped = 0

    def wait(self, n):
        if not self.enabled:
            return 1
        if self.t0 is None:
            self.t0 = time.perf_counter()
        remain = (self.t0 + self.dt * n) - time.perf_counter()
        return max(1, int(remain * 1000)) if remain > 0 else 1


def alert_overlay(vis, active, alert_cfg, first):
    """R4 draw_alert() 로 알림을 그리되, 화면 틴트 색은 config 로 제어한다.

    decision.py 의 draw_alert() 는 두 가지가 고정돼 있다.
      1) 화면 틴트가 붉은색 하드코딩  -> red_alpha=0 으로 끄고 여기서 직접 입힌다
      2) 호출할 때마다 [ALERT] 를 print -> 배너를 계속 띄우려면 매 프레임 불러야
         하는데 그러면 콘솔이 도배된다. 최초 1회만 통과시키고 나머지는 삼킨다.
    둘 다 R4 파일을 수정하지 않고 처리한다.
    """
    if not active:
        return vis

    # 1) 화면 틴트 (야간은 녹색, 낮은 적색 — config.ALERT_OVERRIDE 참조)
    a = alert_cfg.get("tint_alpha", 0.0)
    if a > 0:
        layer = np.full_like(vis, alert_cfg.get("tint_bgr", (0, 0, 255)))
        vis = cv2.addWeighted(vis, 1 - a, layer, a, 0)

    # 2) 테두리·문구·소리·로그는 R4 함수에 맡긴다
    if first:
        return draw_alert(vis, True, alert_cfg, save=False, play_sound=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return draw_alert(vis, True, alert_cfg, save=False, play_sound=False)


def apply_decision_cfg(state, cfg):
    """config.VIDEOS[...]['decision'] 값으로 R4 상태를 덮어쓴다.

    decision.py 의 DECISION_CONFIG 는 day/night 두 가지뿐이라 영상 3종을
    개별 튜닝할 수 없다. R4 파일을 고치는 대신 여기서 주입해서
    '설정은 config.py 한 곳에' 규약을 지킨다.
    """
    for k, v in cfg.get("decision", {}).items():
        setattr(state, k, v)
    return state


def draw_panel(frame, lines, org=(12, 44), scale=0.55):
    """좌상단 정보 패널. R4의 perf HUD 아래에 붙인다."""
    pad, lh = 8, 22
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
             for t, _ in lines) + pad * 2
    th = lh * len(lines) + pad * 2
    x, y = org
    box = frame[y:y + th, x:x + tw]
    if box.size:
        frame[y:y + th, x:x + tw] = cv2.addWeighted(
            box, 0.30, np.zeros_like(box), 0.70, 0)
    for i, (t, c) in enumerate(lines):
        cv2.putText(frame, t, (x + pad, y + pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)


# ------------------------------------------------------------------
# 미션 A · 앞차 출발
# ------------------------------------------------------------------
def run_mission_a(key, cfg, show, save, probe, realtime=False):
    path = find_video(cfg["file"])
    if not path:
        print(f"  ★ 영상 없음: {cfg['file']}")
        return None

    if config.USE_YOLO:
        vehicle_yolo.set_profile(cfg["yolo_profile"])

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None

    pacer = Pacer(fps, realtime and show)
    state = apply_decision_cfg(DecisionState(mode=cfg["mode"]), cfg)
    glare_max = config.GLARE_MAX_RATIO[cfg["mode"]]
    buf = deque(maxlen=cfg["gap"] + 1)
    fired_at, i, stats = None, 0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- R1 : 전처리 (해상도 통일 + 그레이 + 밤 보정 + 블러) ---
        gray, meta = preprocess(frame, mode=cfg["mode"])
        color = meta["color"]                      # 리사이즈만 된 컬러 (YOLO 입력)

        # --- YOLO : 앞차 ROI 자동 설정 (실패해도 폴백으로 계속 돈다) ---
        roi = (vehicle_yolo.get_vehicle_roi(color)
               if config.USE_YOLO else cfg["roi"])

        # --- R2 : 차영상 움직임 검출 ---
        buf.append(gray)
        pixels, mask = 0, None
        if len(buf) > cfg["gap"]:
            pixels, mask = detect_motion(buf[0], buf[-1], mode=cfg["mode"],
                                         roi=roi, return_mask=True)

        x1, y1, x2, y2 = roi_to_px(roi, gray.shape)
        area = max(1, (x2 - x1) * (y2 - y1))
        ratio = pixels / area

        # --- 조명 급변 차단 ---
        # 신호가 바뀌며 화면 전체 색이 변하면 ROI 의 절반 이상이 '움직임'으로
        # 잡힌다. 실제 출발보다 훨씬 크므로 상한을 넘으면 0 으로 눌러버린다.
        glare = ratio > glare_max
        if glare:
            pixels = 0

        stats.append(ratio)

        # --- R4 : 판정 ---
        just_fired = decide(state, {"motion_pixels": pixels})
        if just_fired and fired_at is None:
            fired_at = i

        if probe:
            i += 1
            continue

        # --- 출력 ---
        vis = color.copy()
        if mask is not None and mask.size:
            layer = np.zeros_like(vis[y1:y2, x1:x2])
            layer[mask > 0] = (0, 0, 255)
            vis[y1:y2, x1:x2] = cv2.addWeighted(
                vis[y1:y2, x1:x2], 1.0, layer, 0.35, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(vis, "ROI(YOLO)" if config.USE_YOLO else "ROI",
                    (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 2)

        vis = alert_overlay(vis, state.status == STARTED,
                            config.alert_config(key), just_fired)
        draw_perf_hud(vis)
        draw_panel(vis, [
            (f"{cfg['label']}  [MISSION A]", (255, 255, 255)),
            (f"frame {i}  {i / fps:.2f}s", (200, 200, 200)),
            (f"motion {pixels:6d}px  ratio {ratio:.4f}", (200, 200, 200)),
            (f"thr {state.motion_pixel_threshold}px  x{state.consecutive_frames_required}",
             (150, 150, 150)),
            (f"STATE: {'조명변화 무시' if glare else state.status}",
             (0, 200, 255) if glare else
             (0, 255, 90) if state.status == STARTED else (255, 255, 255)),
        ])

        if save:
            if writer is None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                writer = cv2.VideoWriter(
                    os.path.join(OUTPUT_DIR, f"{key}_result.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if show:
            cv2.imshow("Start-Alert", vis)
            pacer.shown += 1
            if cv2.waitKey(pacer.wait(i)) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return fired_at, i, fps, stats, pacer


# ------------------------------------------------------------------
# 미션 B · 신호등
# ------------------------------------------------------------------
def run_mission_b(key, cfg, show, save, probe, realtime=False):
    path = find_video(cfg["file"])
    if not path:
        print(f"  ★ 영상 없음: {cfg['file']}")
        return None

    if config.USE_YOLO:
        vehicle_yolo.set_profile(cfg["yolo_profile"])

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None

    pacer = Pacer(fps, realtime and show)
    state = apply_decision_cfg(DecisionState(mode=cfg["mode"]), cfg)
    fired_at, i = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray, meta = preprocess(frame, mode=cfg["mode"])
        color = meta["color"]

        # --- R3 : 신호등 색 판정 (ROI 는 YOLO 가 제안, 없으면 R3 config) ---
        override = None
        if config.USE_YOLO:
            lroi = vehicle_yolo.get_light_roi(color)
            if lroi:
                override = {"roi": lroi}
        light, ev = detect_light(color, profile=cfg["light_profile"],
                                 light_config=override, return_evidence=True)

        # --- R4 : 판정 (light_color 가 있으면 미션 B 경로로 분기) ---
        just_fired = decide(state, {"light_color": light, "side_clear": True})
        if just_fired and fired_at is None:
            fired_at = i

        if probe:
            i += 1
            continue

        vis = color.copy()
        lx, ly, lw, lh = ev.roi
        lc = ((0, 0, 255) if light == "RED" else
              (0, 255, 0) if light == "GREEN" else (150, 150, 150))
        cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), lc, 2)
        cv2.putText(vis, f"LIGHT {light}", (lx, max(16, ly - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)

        vis = alert_overlay(vis, state.status == STARTED,
                            config.alert_config(key), just_fired)
        draw_perf_hud(vis)
        draw_panel(vis, [
            (f"{cfg['label']}  [MISSION B]", (255, 255, 255)),
            (f"frame {i}  {i / fps:.2f}s", (200, 200, 200)),
            (f"light: {light}", lc),
            (f"red {ev.red_area:5d}  green {ev.green_area:5d}", (200, 200, 200)),
            (f"STATE: {state.status}",
             (0, 255, 90) if state.status == STARTED else (255, 255, 255)),
        ])

        if save:
            if writer is None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                writer = cv2.VideoWriter(
                    os.path.join(OUTPUT_DIR, f"{key}_result.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if show:
            cv2.imshow("Start-Alert", vis)
            pacer.shown += 1
            if cv2.waitKey(pacer.wait(i)) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return fired_at, i, fps, None, pacer


# ------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    show = "--show" in args
    probe = "--probe" in args
    realtime = "--realtime" in args
    save = "--no-save" not in args and not probe
    keys = [a for a in args if not a.startswith("--")] or list(config.VIDEOS)
    t_all = time.perf_counter()

    print(f"YOLO: {'ON' if config.USE_YOLO else 'OFF'}"
          f"{' (모델 없음 -> 수동 ROI)' if config.USE_YOLO and not vehicle_yolo.available() else ''}")
    print(f"\n{'영상':<12} {'발화':>8} {'시각':>9} {'정답':>7} {'오차':>9}")
    print("-" * 50)

    for key in keys:
        cfg = config.VIDEOS.get(key)
        if not cfg:
            print(f"  모르는 이름: {key} (가능: {list(config.VIDEOS)})")
            continue
        fn = run_mission_a if cfg["mission"] == "A" else run_mission_b
        t0 = time.perf_counter()
        r = fn(key, cfg, show, save, probe, realtime)
        dt = time.perf_counter() - t0
        if not r:
            continue
        fired, total, fps, stats, pacer = r
        gt = cfg["ground_truth"]

        if probe and stats:
            a = np.array(stats)
            still = a[max(0, gt - 300):max(1, gt - 20)]
            move = a[gt:min(len(a), gt + 120)]
            print(f"{key:<12} ratio  정차 평균={still.mean():.5f} 최대={still.max():.5f}"
                  f" | 출발 평균={move.mean():.5f} 최대={move.max():.5f}")
            continue

        src_len = total / fps
        speed = src_len / dt if dt else 0
        late = (f"  표시 {pacer.shown}/{total}f" if realtime and pacer.dropped
                else "")
        if fired is None:
            print(f"{key:<12} {'미탐':>8} {'-':>9} {gt:>7} {'-':>9}"
                  f"  | {dt:5.1f}s / 원본 {src_len:5.1f}s  {speed:4.2f}x{late}")
        else:
            print(f"{key:<12} {fired:>8} {fired / fps:>8.2f}s {gt:>7} "
                  f"{(fired - gt) / fps:>+8.2f}s"
                  f"  | {dt:5.1f}s / 원본 {src_len:5.1f}s  {speed:4.2f}x{late}")

    if not probe:
        print(f"\n총 소요 {time.perf_counter() - t_all:.1f}초")
    if save:
        print(f"결과 영상 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()