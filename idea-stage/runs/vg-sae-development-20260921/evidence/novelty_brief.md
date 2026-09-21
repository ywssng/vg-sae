# 현재 VG-SAE 방법의 신규성 검토용 정의

사용자가 고정한 연구는 Variational Garrote 기반 새 SAE 개발이다.
`RESEARCH_BRIEF.md`가 원래 목표와 이번 실행의 범위를 규정한다.
현재 방법을 별개 진단 연구로 교체하는 것은 범위 이탈이다.

## 방법과 주장 대상

입력 x에서 두 affine head로 m=sigmoid(g(x)), a=softplus(r(x))를 구하고,
unit-norm 선형 dictionary D로 복원한다. support q(s|x,a(x),D)는 independent
Bernoulli 변분 분포로 취급하고 amplitude는 점 추정한다.

E=.5||x-b-D(ma)||²+.5Σm(1−m)a²||D_j||².
정규화된 Bernoulli prior와 entropy weight1의 차이는 KL(q(s)||p(s))이고,
full objective는 beta E −d/2 log(beta/(2pi))+KL이다.
학습에는 latent sampling 또는 hard-threshold STE가 필요하지 않다.
배포는 native `1[m>.5]*a`; mean 및 sampled reconstruction은 별도 관측치다.

learned beta는 global log precision을 gradient로 학습한다.
profiled 구현은 minibatch energy를 사용한다. 두 stochastic optimization 문제를
동일하다고 주장하지 않는다. a(x) 자체의 prior/entropy가 없으므로 일반적인
spike-and-slab 생성모델 전체의 ELBO나 calibrated semantic posterior라고 주장하지 않는다.

## 검증할 기여

1. 원래 VG의 선택·진폭·noise coupling을 입력별 amortized SAE와 learned dictionary로
   옮긴 구체적 방법 조합이 기존 방법과 같은가? 학습/추론/precision 차이는 무엇인가?
2. 기존 Gated/JumpReLU/TopK 계열과 변분 sparse coding 대비 어디에서 support 회복,
   amplitude bias 또는 sparse reconstruction의 추가 이점이 가능한가?
   성능 이점은 아직 검증 대상이며 방법 구성만으로 성공을 선언하지 않는다.
3. analytic expectation은 어떤 sampled 방법의 어떤 계산을 줄이는가? 단순히 모든
   variational 방법보다 빠르다는 넓은 주장은 하지 않는다.

고정된 main method의 신규성·강점·필수 근거를 판정하되, 작은 optimizer 초기화 팁이나
기존 coordinate update 자체를 새 주방법의 기여로 대신하지 않는다.

## 기존 근거와 새로운 파일럿

`pipeline_audit.md`/`pipeline_evidence.json`에 기존 전체 곡선을 보존했다.
Stage1 exponential 조건의 낮은 latent error는 탐색 신호지만 test-curve 최저점과
서로 다른 실제 L0라는 한계가 있다. Stage2 일부낮은L0에서 mean-hard gap은작다.
Stage3 완료 checkpoint 일부가 있으나 평가가 없어 실제 activation 우위를 주장할 수 없다.

P1은 원학습 dictionary를 복원검증한 뒤 독립cal/test에서 기존checkpoint를 고른다.
P2는 learned beta=1/train-moment initialization과 minibatch-profiled reference를
36개 작은 학습으로 비교한다. P3는 conditional VG coordinate correction의
품질·sparsity·latency와 공통 refit control을 검사한다. 모두 진행 중이며 결과를
신규성 판정에 유리하게 만들어 넣지 않는다.

## 근접 선행

`literature_update.md`, `literature_sources.json`의32개 primary references를 사용한다.
특히 원래VG, Soh2025VG, Gated, JumpReLU, ProbabilisticTopK, Baker–Li vSAE,
vsPAIR, Velychko entropyELBO, Lu et al. VAEase ICML2025, Tonolini VSC,
Fallah–Rozell thresholded VSC, Geadah SVAE를 objective 수준에서 비교한다.
선행연구가 근처에 있다는 이유만으로 연구 전체를 포기하지 않는다.
반대로 현 조합/주장을 그대로 포함한 named paper가 있으면 어떤 claim이 중복인지 명확히 쓴다.
