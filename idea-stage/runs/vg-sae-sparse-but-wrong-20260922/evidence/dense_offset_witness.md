# Dense affine offset의 구성 예 — 학습 결과의 인과 설명과 구분

새 training pilot이 아니라, 현재 VG 함수에 실제로 넣을 수 있는 parameter family를
정의해 objective와 readout의 관계를 확인한 deterministic 수식 점검이다.
어떤 실제 SGD run이 이 경로를 따랐다는 증거로 쓰지 않는다. 독창적인 정리도 주장하지 않는다.

Orthonormal D, x=Dz, z≥0와 finite second moment를 가정한다. t≥0에 대해
`b_t=−t D 1`, `W_a=D^T`, `c_a=0`, `W_g=0`, `c_g=(4t+4)1`로 둔다.
그러면 `a_t=softplus(z+t)`, `m_t=sigmoid(4t+4)`다.
`softplus(z+t)−(z+t)≤exp(−t)`이며 `1−m_t≤exp(−4t−4)`다.
Mean residual risk는0으로가고 Bernoulli variance도 `O((t²+E||z||²)exp(−4t))`로0에간다.
Hard readout은모든좌표에서양수여서L0=J지만 D는계속정답이고 c_dec=0이다.

고정 finite gamma에서 support KL은 `J*softplus(gamma)`로수렴해유한하다.
따라서 **epsilon이없는 이상적인 profiled objective**의 `d/2*log(2E/d)+KL`은
이경로에서−infinity로간다. 같은energy에 beta를전역최적화하는이상식도동일한문제를갖는다.
**Profiled branch**는 `loss_eps`로 energy/log를 clamp하므로 해당 구현이 그대로 무한히 낮아진다고 표현하지 않는다.
**Global learned-beta branch**는 energy를 이 epsilon으로 clamp하지 않는다. 그 branch까지
epsilon이 유한 하한을 준다고 일반화하지 않는다. 유한 부동소수점과 overflow는 수학적 하한이 아니다.
제공한수치범위는설정한floor보다위에있으며production epsilon1e−8을쓰는별도test도있다.

고정 finite beta에서는 `beta E+KL−d/2*log(beta/(2pi))`의앞두항이음수가아니므로
유한한Gaussian상수하한이있다. 이는이특정precision-driven 발산을없애는성질이며,
바른feature/support를배우거나모든dense해를제거한다는보장은아니다.
두mode는서로다른상수항/학습절차이므로rawloss값의cross-mode우열을비교하지않는다.

이관찰은같은VGarchitecture에서precision정책을분리해보는최소후속의근거다.
현재P1의높은EV/densehardcode와정합적일수있으나그run의원인으로확정하지않는다.
참조논문의feature-mixing정리를VG에그대로옮긴것도아니다.

수치:[dense_offset_witness.json](dense_offset_witness.json).
Regression checks:`tests/test_sbw_dense_offset.py`.
