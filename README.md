# OpenCV_Start_Alert
Start-Alert System using OpenCV for lead vehicle movement and traffic light detection.

## 디렉토리 구조

역할(R1~R6)과 문서의 함수 인터페이스에 맞춰 구성한 구조입니다.

```
OpenCV_Start_Alert/
├── requirements.txt        # opencv-python, numpy
├── .gitignore               # videos/, outputs/, __pycache__ 등 제외
├── videos/                  # 제공 영상 3종 (용량 커서 git 미포함)
│   ├── Day_Car.mp4
│   ├── Nigt_Car.mp4
│   └── Night_Nocar.mp4
├── outputs/                 # 실행 결과 영상·캡처·로그 (git 미포함)
├── main.py                  # R5 — 전체 파이프라인 조립, 영상 선택 실행
├── config.py                # R5(+R6) — CONFIG 한 곳에 모음: ROI, 임계값, HSV 범위,
│                             #           영상별(day/night) 프로필 분리
└── src/
    ├── preprocess.py         # R1 — preprocess(frame) → gray/blur/resize, ROI 슬라이싱
    ├── motion.py              # R2 — detect_motion(prev, cur) → 움직임 화소 수 (미션 A)
    ├── light.py                # R3 — detect_light(frame) → 'RED'/'GREEN'/None (미션 B)
    ├── decision.py             # R4 — decide(state, evidence), draw_alert() : 상태 전이·
    │                            #      N프레임 연속 판정·오버레이/소리/로그 출력
    └── night_tuning.py         # R6(6인 팀) — 영상 2·3 전용 파라미터 실험, 오탐·미탐 사례 기록
```

- **역할 = 파일 1:1 매핑**: R1~R4는 각자 파일 하나만 책임지므로 서로 코드 완성을 기다리지 않고 병렬 개발이 가능합니다 (더미 함수부터 시작).
- **config.py로 설정 통합**: ROI·임계값·HSV 범위를 코드 곳곳에 하드코딩하지 않고, `video1_day` / `video2_night` / `video3_night_signal` 프로필로 분리해 `main.py`에서 선택해 사용합니다.
- **videos/, outputs/는 데이터 폴더**: 용량 큰 원본 영상과 실행 결과물은 `.gitignore`로 제외해 레포를 가볍게 유지합니다.
- **night_tuning.py는 선택**: 5인 팀이면 생략하고 R2·R3가 야간 파라미터를 직접 관리해도 됩니다.
