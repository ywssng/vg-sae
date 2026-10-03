# Phase 1–2 문헌 반영 실행 추적

**2026-10-03 현재 상태.** 이관된 W1 및 후속 진단·증명·음성 결과의 전체 목록은 [연구 상태 감사](runs/vg-sae-colab-continuation-20261003/RESEARCH_STATE_AUDIT.md)에 있다. Colab CPU에서 controller 47개와 새 수리/provenance 테스트 51개가 통과했고 결과를 회수했다. 새 terminal audit와 finite-noise 계산은 진행 중이다. BR5 `measurement_unresolved`, W2 gated, C2a/C2b unentered를 유지한다. 아래 표는 이전 실행 이력으로 보존한다.

2026-09-30. 문헌·토론과 계획 개정만 완료했다. 아래 신규 실험은 전부 미실행이다. `EXISTING_DONE`은9월24일 근거이며 이번에 다시 실행했다는 뜻이 아니다.

| ID | 목적 | 새 학습 크기 | 우선순위/상태 | 근거·조건 |
|---|---|---:|---|---|
| P1-B1–B4 | 구현·exact/MF·frozen·joint 근거 | 기존135cells,9+54fits | EXISTING_DONE | 9월24일 run/config/results 보존 |
| P1-P | Gram interaction·covariance·stable MF response 해석/재평가 | 0fits | TODO, 먼저 | 기존exact response를새 결과로세지않음 |
| P1-R | Ambient rotation과stable-correct/incorrect 대조 | 0fits | TODO, 먼저 | fixed-model/function-preserving transformation |
| P1-F | Orthogonal analytic gate parameter witness | 0fits | TODO | 필요시기존3fit continuation은별도 선택 |
| P1-PRIOR | Geometry×co-firing matched/mismatched oracle | 24target evaluations,0fits | SHOULD, TODO_CONDITIONAL | 첫density pilot착수 gate가아님 |
| W1a | E_opt full-curve horizon | 136–200fits, 전부 8k | TODO, 첫학습 | teachers31000/31004, reference+3repeats |
| W1b | E_joint challenger, reference공유 | 102–150추가fits, 전부 8k | TODO, 같은bridge | data+optimizer 결합변동, 순수 data variance아님 |
| BR-READOUT | Soft/hard·axis·reference sensitivity | 0추가fits | TODO, 모든bridge결과보고 | savedmodels, 각native density로별도matching |
| W2 | 나머지개발6teachers | 408–600fits, H_star | TODO_CONDITIONAL | bridge결과뒤명시된protocol lock |
| W3 | 본검증36freshteacher-worlds | 3672–5400fits, H_star | TODO_CONDITIONAL | primary/version잠금후truth-blindprediction |
| W4 | SmallL1 generic-stability 비교 | 612–900fits, H_star | TODO_CONDITIONAL | W3진입시사전포함, 결과 무관보고 |
| X1 | 세objective/precision추가methods | 1836–2700fits | OPTIONAL_CONDITIONAL | 질문·subset·endpoint를추가 학습전고정 |
| X2 | TopK recovery-family 비교 | 576fits | OPTIONAL_CONDITIONAL | nativecurve ineligibility를VG우위로세지않음 |
| X3 | 분포·geometry transfer | 918–1350fits | OPTIONAL_CONDITIONAL | core범위를넓힐때새근거 |
| Stage2/3 | SynthSAEBench/LM 외적검증 | 별도계획 | NOT_SCHEDULED | freshstreams·비용·모형범위확인 |

W1a+b는238/350 distinctfits이고4k-step equivalents476/700이다. 2k/4k/8k snapshot을새 fit으로세지않는다. W2–W4까지모두진행하면total4930/7250fits. H_star4000이면5168/7600equivalents,8000이면9860/14500equivalents다. 같은 teacher의dataset draws·optimizer repeats·sharedreference는독립 world n이아니다.

문헌 검토의완료가실험의성공이나새GPU 실행승인은아니다. 비용과숫자threshold는제안이며실제 timing과개발후확정한다. 미래구현이원래9월29일budget를그대로자동실행하지않도록현재계획의W0/W1 gate를먼저읽는다.
