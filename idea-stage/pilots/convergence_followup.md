# 수렴 확인을 위한 후속 파일럿

2026-09-19. 최초 1200-step 파일럿과 비판 리뷰를 본 뒤 계획한 탐색 분석이다.
기존 사전 계획의 성공 기준이나 결과를 덮어쓰지 않는다.

계기: 마지막 약 200 step에도 beta가 35–42% 증가하고 loss가 감소했다.
따라서 width와 prior control 차이가 단순 학습 지연인지 점검한다.

- 같은 데이터/optimizer seed 0,1과 LR/batch, widths32/128을 사용한다.
- 최초부터 6000 steps를 학습하며 모든 다른 설정은 유지한다.
- width32 fixed-pi는 두 prior rule의 공통 anchor, width128에서 두 rule 비교.
- 결과를 `outputs/idea_discovery_20260919/convergence/`에 분리 보관한다.
- 마지막 두 history points의 상대 beta 변화와 loss 변화, mean/sample/hard
  reconstruction EV를 함께 읽는다. 작은 변화 자체도 전역 최적성 증거는 아니다.
- mean EV에서 stochastic variance 항을 뺀 sample EV를 추가한다.
  posterior-mean 복원과 E_q squared reconstruction risk를 혼동하지 않는다.
- 6000에서도 width gap/negative simple fix가 남으면 budget 확대에 대한
  민감도 확인으로만 보고하고, 남지 않으면 최초 관찰의 범위를 축소한다.
- 추가 GPU 예산 추정 0.1시간, timeout 15분/seed.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/idea_discovery_training_pilot.py --seed 0 --device cuda:0 --widths 32 128 --steps 6000 --output-dir outputs/idea_discovery_20260919/convergence
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/idea_discovery_training_pilot.py --seed 1 --device cuda:1 --widths 32 128 --steps 6000 --output-dir outputs/idea_discovery_20260919/convergence
```
