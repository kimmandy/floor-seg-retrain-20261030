# 2026-10-29 kmj 학습 수정 (bok ratio811 설정 사본)

- B1 박스 확장 끔: train_all6.py min_training_box_side 20 → 0 (작은 물체 박스가 마스크보다 커서 이웃과 겹침)
- B2 마스크 밀림 패치: mask_resize_fix.install() (handoff_20261028_iou50) — 정답 +0.7 px 밀림 제거
- C 밀집 장면: all6_sampling COUNTS 에 fastener_crowd 280 추가, 데이터셋 /data/work/datasets/ratio811_crowd_20261029 (crowd_paste.py)
- 경로: paths.json · start_training.sh → kmji. 공용 잠금은 bok 과 같음 (bok 학습 중이면 시작 거부)
- 초기 가중치·valid·test·평가 지표·선택 기준은 bok ratio811 과 같음 → 같은 조건 비교 가능
