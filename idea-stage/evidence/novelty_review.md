# VG-SAE fresh novelty review

- 검토일: 2026-09-19
- reviewer identity: `/root/novelty_ultra`; model `gpt-6-astra`; reasoning effort `ultra`
- route/status: fresh task agent; **same-family provisional**
- 입력: [11개 후보](brainstorm_candidates.json), [code audit](code_audit.md), [28개 canonical 출처와 1개 lead](literature_sources.json), 현재 root pilot 결과.
- 방법: 주요 주장 4개마다 5개 이상 다른 query, 2026-03-19–09-19 날짜범위 arXiv API 검색, 직접 겹치는 수식·방법의 primary body 확인. 구체적인 검색은 [novelty_sources.json](novelty_sources.json).
- 독립성: 초기 후보를 생성한 에이전트와 다른 agent다. root가 제시한 Cavazza 선행과 코드 관찰을 알고 시작했으며, blind 심사는 아니다. 해당 선행은 원문에서 재검증했다.

## 판정

**PROCEED WITH CAUTION — 5/10.** 허용되는 범위는 **현재 VG-SAE에서 gate 확률의 대상과 posterior-mean / sampled / hard readout을 분리하고, width에 따른 변화의 원인을 통제 실험으로 밝히는 진단 연구**다. 독립적인 방법 논문이나 새로운 replication 정리가 이미 확보된 것은 아니다.

차이가 얇은 구체적인 이유는 세 가지다. 복제에 따른 variance dilution 및 width-scaled retain probability는 Cavazza 외(2018)에 있다. 작은 soft gate의 정보 누출과 hardening은 SoftSAE(2026)에 있다. 현재 학습 pilot은 전자의 메커니즘이 실제로 발생했다는 인과 증거를 제공하지 않으며, 단순 prior/readout 처방도 성공하지 않았다. 따라서 'VG가 발견한 새 보편적 실패'를 중심 claim으로 올리면 안 된다.

검증 가능한 delta 문장: **“입력 의존 amplitude를 가진 VG-SAE에서 gate posterior의 prior floor, 평균 코드·확률적 코드·경성 코드의 reconstruction, 그리고 conditional support inference를 분리하여, 기존 soft-reconstruction 이득이 어떤 확률적·희소 표현 주장을 실제로 지지하는지 판별한다.”** 이는 연구 질문과 목표 기여이며, 현재 pilot이 이미 모든 항목을 입증했다는 뜻은 아니다.

B05의 **known generative support posterior → mean-field → amortization → learned-amplitude target** 분리 질문은 **PROCEED — 6/10**이다. 이름 붙인 동일 연구를 찾지 못했고, 가까운 probabilistic SAE와 구분 가능한 검증 대상이 있다. 다만 알려진 variational inference 현상을 VG toy에서 재현하는 데 그치면 논문 기여는 작다.

## 4개 주장과 선행 경계

| 주장 | 가장 가까운 선행 | 알려진 부분 | 남아 있는 범위와 판정 |
| --- | --- | --- | --- |
| C1: prior-matched gate·복제 amplitude가 threshold와 양립하지 않는 해를 허용 | [Cavazza 2018](https://proceedings.mlr.press/v84/cavazza18a/cavazza18a.pdf), [SoftSAE 2026](https://arxiv.org/html/2605.06610v1) | 열 복제 1/r variance, 무한폭 infimum, 폭별 retain probability; tiny soft weights와 hard stage | 실제 VG 학습에서 발생 조건·정도·원인을 입증하면 차이. PROCEED WITH CAUTION |
| C2: support probability 의미와 보정 검증 | [Probabilistic TopK](https://openreview.net/pdf?id=zMIIHeKivz), [vSAE](https://arxiv.org/abs/2509.22994), [vsPAIR](https://arxiv.org/abs/2602.02948) | 확률적 SAE 및 uncertainty 출력 | 정확한 생성 support와 conditional amplitude target을 분리하는 controlled study. PROCEED |
| C3: dictionary-aware sparse readout | [DSS](https://arxiv.org/html/1408.0464), [MPM](https://arxiv.org/abs/1807.08336), [Gated SAE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html) | posterior와 sparse decision 분리; 0.5 threshold의 제한; support/magnitude 분리 | 같은 L0·비용의 causal attribution 또는 VG 특유 이득이 필요. PROCEED WITH CAUTION |
| C4: frozen VG gate refinement | [O'Neill 2025](https://proceedings.mlr.press/v267/o-neill25a.html), [원래 VG](https://doi.org/10.1007/s10994-013-5427-7), [LocA-SAE 코드](https://github.com/wenjie1835/Local_Amotized_SAEs) | SAE amortization gap, iterative sparse inference, VG coordinate update | 동일 objective에서 gate-only refinement와 hard feature quality의 방향을 분리. PROCEED WITH CAUTION |

**좁은 ABANDON 판정:** “복제하면 Bernoulli variance penalty가 1/r로 작아지고, retain probability를 폭에 따라 줄이면 이를 막는다”를 새 일반 정리로 내세우는 scope는 **ABANDON**이다. 명명한 published paper는 **Cavazza et al., AISTATS 2018, §4 Eq. 13/Proposition 1, §5 Eq. 15**다. 이는 B04의 VG-specific empirical study 전체를 버리라는 판정이 아니다. 같은 공간에 이웃이 있다는 이유로 다른 후보를 ABANDON하지 않았다.

Cavazza의 variance algebra에 VG prior의 zero-KL과 threshold output=0을 대입하는 것은 현재 구현에 유용한 construction이다. 추가 결론이 코드의 보장 부재를 정확히 드러내지만, algebra의 간단한 특수화만으로 강한 독립 이론 기여를 주장하기 어렵다. learned amplitudes가 입력 정보를 운반하므로 gate KL=0이 전체 code의 정보량=0을 뜻하지 않는다는 사실도, 그 자체로 calibrated semantic uncertainty의 실패를 증명하지 않는다.

## 전체 11개 후보의 jury

점수는 10점 만점이다. 정보량은 pilot이 다음 결정을 얼마나 바꾸는지, upside는 성립했을 때 연구 기여의 크기, repo fit은 현재 코드·데이터로 검증하기 좋은 정도다. 점수의 단순 합으로 순위를 결정하지 않았다. 알려진 control은 정보량이 높아도 독립 기여 순위가 낮을 수 있다.

| 순위 | 후보 | 정보량 | upside | repo fit | novelty | 판정 | 이유 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B04 | 9 | 8 | 10 | 5 | PROCEED WITH CAUTION | Mechanism-plus-occurrence package; exact algebra not new; includes B01 and B08 as controls. |
| 2 | B05 | 9 | 8 | 8 | 6 | PROCEED | Exact posterior reference and target separation supply highest independent scientific upside. |
| 3 | B07 | 8 | 6 | 9 | 5 | PROCEED WITH CAUTION | Directional diagnostic for unchanged VG objective; generic amortization gap already published. |
| 4 | B06 | 6 | 8 | 6 | 5 | PROCEED WITH CAUTION | High upside if a specific competition failure is repaired; oracle pairs and general structured VI are not a finished SAE method. |
| 5 | B02 | 7 | 6 | 9 | 4 | PROCEED WITH CAUTION | Sparse posterior projection is known; must beat pursuit/refit and expose a VG-specific phenomenon. |
| 6 | B11 | 7 | 6 | 8 | 5 | PROCEED WITH CAUTION | Useful variance/precision feedback diagnostic; real-activation noise target is unavailable. |
| 7 | B09 | 6 | 6 | 7 | 4 | PROCEED WITH CAUTION | Rare-feature empirical Bayes can reveal feedback failure; prior adaptation itself standard. |
| 8 | B10 | 5 | 6 | 6 | 4 | PROCEED WITH CAUTION | Beta-Bernoulli adaptive counts face BatchTopK, SoftSAE and sparse attention competition. |
| 9 | B01 | 10 | 5 | 10 | 4 | PROCEED WITH CAUTION | Very useful supporting intervention; unlikely independent paper contribution without consequential finding. |
| 10 | B03 | 9 | 4 | 10 | 3 | PROCEED WITH CAUTION | Cheap necessary attribution control, not a new NNLS or amplitude method. |
| 11 | B08 | 8 | 5 | 10 | 3 | PROCEED WITH CAUTION | Necessary width/prior control; standalone rule is near dropout scaling and multiplicity priors. |

결과를 받기 **전** root에 전달한 상위 3개는 **B04, B05, B07**이다. B04는 B01의 grouped intervention과 B08의 width-prior control을 포함한 하나의 인과 진단 묶음이다. B03은 저렴한 필수 readout control로 포함할 가치가 있다. 따라서 root가 실행한 B04 construction, B08 width sweep, B01/B03 readout package는 서로 독립적인 새 아이디어 세 개보다 **B04 메커니즘 후보를 값싸게 분해하는 묶음**으로 보는 것이 정확하다. B05는 그 결과가 부정적일 때도 남는 독립 질문이다. B06은 방법 upside가 높지만 oracle pair의 성공에서 학습된 dictionary의 안정된 grouping으로 넘어가는 비용이 커 우선순위 4다.

## 현재 pilot이 verdict를 어떻게 제한하는가

이 절은 jury 이후 받은 결과를 반영한다. reviewer가 새 training을 실행한 것은 아니다. 아래 raw JSON을 읽고 stochastic EV를 직접 재계산했다.

- `outputs/idea_discovery_20260919/replication/results.json`
- `outputs/idea_discovery_20260919/training/seed0/results.json`
- `outputs/idea_discovery_20260919/training/seed1/results.json`
- `outputs/idea_discovery_20260919/posterior/results.json`

### 세 가지 reconstruction은 다르다

중심화와 동일 denominator를 사용하면,

\[
R_{\mathrm{mean}}=\|x-D(m\odot a)\|^2,\quad
R_{\mathrm{sample}}=R_{\mathrm{mean}}+\sum_jm_j(1-m_j)a_j^2\|d_j\|^2,
\]
\[
R_{\mathrm{hard}}=\|x-D(\mathbf1[m>.5]\odot a)\|^2.
\]

현재 JSON의 `expected_ev`는 **posterior-mean reconstruction EV**다. 실제 stochastic expected reconstruction EV는
`expected_ev - 2*variance_energy/test_centered_energy`이다. Gaussian loss가 최적화하는 reconstruction 항은 후자에 대응한다.

폭·prior·seed의 **10개 학습 configuration 모두 hard EV가 sampled EV보다 높았다**. 예를 들어 fixed-pi 폭128에서 seed0은 mean/hard/sample EV=0.90141/0.85103/0.84200, seed1은 0.90314/0.85234/0.84157이다. 따라서 mean EV가 hard EV보다 높다는 관찰만으로 “학습 objective가 경성 inference에 실패한다”는 주장은 지지되지 않는다. posterior 평균이 좋은 point predictor인 현상과 학습된 objective의 보장 실패를 구분해야 한다.

### 복제 경로와 실제 학습 결과는 구분해야 한다

구성 실험은 구현에서 정확한 1/r와 hard-zero를 확인한다. 그러나 fixed-pi learned width32→128에서 variance energy는 seed0 **0.09581→0.11504**, seed1 **0.09725→0.11920**으로 증가한다. decoder 유사도 및 mean–hard gap 증가만으로 학습된 모델이 구성의 variance-dilution 경로를 이용했다고 결론낼 수 없다. 1,200-step fixed-budget 결과는 최적화 상태도 별도 통제해야 한다.

fixed-count prior는 폭128에서 hard EV가 fixed-pi보다 낮고, 현재 tested same-support mean/top-ma readout도 rescue를 보이지 않았다. 이는 “width-scaled prior가 해결책”이나 “ma로 읽으면 해결”이라는 가설의 부정 증거다. tested readout 결과를 아직 실행하지 않은 수렴된 NNLS나 exhaustive sparse action 전체의 실패로 확장하면 안 된다.

root는 near-prior group 제거 효과 중 상당 부분이 train-mean replacement에서 줄어 bias 역할일 수 있다고 추가 보고했다. 해당 group ablation의 raw file은 이 reviewer가 독립 재계산하지 않았으므로 root 결과로만 취급한다. future claim에는 train-only bias replacement와 matched-count random groups를 포함해야 한다.

### Exact-posterior pilot

known dictionary·amplitude·beta에서 orthogonal refined mean-field가 exact posterior에 수치적으로 일치하는 control은 유용하다. 하지만 orthogonal exact posterior는 애초에 linear-sigmoid encoder로 표현 가능하다. beta=25, amplitude=1이면 대각 weight=25와 bias=logit(0.15)−12.5≈−14.2346이 정답인데, pilot의 trained weight≈4.9와 bias≈−3.5는 아직 거리가 크다. 따라서 그 조건의 refinement gain을 architecture의 표현력 부족으로 설명하면 안 되며, finite-budget optimization과 표본 효과를 먼저 분리해야 한다.

coherent095에서는 refinement가 Brier 0.03335→0.01503과 free energy를 개선하지만, **marginal NLL은 0.13069→0.16030으로 악화**한다(exact 0.03657). “보정 상태 전반이 개선됐다”는 문장은 틀리다. mean-field의 posterior dependence 누락, reverse-KL 방향, posterior target을 분리하는 좋은 진단이지만 그 일반 현상은 알려진 variational-inference 성질이다. 동일 fixed model을 넘어 learned-amplitude VG에서 의미 있는 차이를 밝힐 때 기여가 커진다.

## 다음 단계에서 주장할 수 있는 기준

1. B04를 메인으로 유지하려면 construction의 알려진 대수와 실제 발생을 분리하고, **수렴·seed·데이터 조건을 바꿔도 남는 VG-specific causal interaction**이 필요하다. 작은 replica intervention으로 mechanism을 직접 움직이되, mean/sample/hard risks 및 feature recovery를 함께 보고한다.
2. B05는 known-model control이 이미 가능한 코드에 잘 맞는다. 같은 encoder가 exact solution을 표현하는 조건을 optimization control로 쓰고, correlated alternatives와 learned a(x)에서 probability target이 달라지는 구간을 조사한다. proper score를 하나만 골라 성공 판정하지 않는다.
3. B07은 no-extra-parameters inference control로서 유용하다. 충분히 수렴한 coordinate objective와 compute-matched refit을 비교해 generic sparse inference 이득을 VG 고유 현상으로 포장하지 않는다.
4. 현재 결과만으로 probabilistic SAE 일반 실패, VG uncertainty superiority, learned replica mechanism 발생, new automatic-L0 criterion, full real-activation 우위를 주장하지 않는다.

이 검토는 파일럿을 계속할 근거를 제공한다. **현재 근거의 가장 강한 결과는 과도한 신규성·uncertainty·objective-mismatch 해석을 제거하고 서로 다른 원인을 판별 가능한 질문으로 바꾼 것**이다. 새 방법의 유효성이나 상위 학회 수준의 완성된 기여는 아직 검증되지 않았다.

## 적용한 novelty verdict limits — 원문

```text
=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
```
