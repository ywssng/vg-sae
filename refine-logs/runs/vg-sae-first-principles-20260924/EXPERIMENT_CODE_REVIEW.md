# 실행 전 코드 검토

대상: `scripts/run_first_principles.py`, `src/sae_inference.py`, `configs/first_principles_20260924.json`, 이번 EXPERIMENT_PLAN.

## 실제 reviewer 응답

- `/root/vg_implementation_audit`, GPT-6 Astra ultra: **SAFE TO RUN**, BLOCKING 없음. Exact normalized likelihood/prior, factorized energy, coordinate update, data split, frozen target, truth evaluation, no test selection을 확인했다. 자신이 작성하지 않은 새 inference module과 runner를 검토했다.
- `/root/baseline_audit`, GPT-6 Astra ultra: B4 baseline constructor, native inference, same-width signed Hungarian matching, dtype/device, config export에 BLOCKING 없음.
- `review_independence: same-family`, `acceptance_status: provisional`. 다른 모델 계열의 승인이나 사람의 심사를 뜻하지 않는다.

## 지적 후 반영

1. B2에서 계산했던 negative log evidence, Var(N), pair covariance를 CSV/NPZ에 보존했다.
2. B3 encoder fixed-point residual을 sample별·평균·최대로 추가했다.
3. Config의2GPUh 상한을 worker별 budget으로 배분하고 fit·update에서 검사한다. 외부timeout을 함께 적용한다. 성공/실패 worker의 wall time을 기록한다.
4. Profile Jensen inequality는 floor_active=False에만 해석한다.
5. Baseline variant와 trainer recipe, 설치 SAELens commit, 현재 git revision/dirty 상태를 기록한다. 실행 후 source와 결과를 함께 커밋한다.

Delta 재검토도 SAFE TO RUN. Exact smoke에서 orthogonal max marginal error8.88e−16, response residual<4e−10, free-energy identity residual<2.1e−15를 확인했다. Coherence.95 smoke의 KL은 본 결과의 근거로 사용하지 않는다.

GPU smoke는 frozen2 fits와 joint6 fits 각각30 steps를 오류 없이 완료했다. Warm step 속도는 2000 updates당 약8–17초 예상이며 초기CUDA/optimizer setup의 일회 비용은 별도다. 이 수치는 예산 추정용이고 본 학습의 수렴 여부를 의미하지 않는다.
