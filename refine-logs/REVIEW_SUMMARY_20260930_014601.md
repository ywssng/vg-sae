# 문헌·두 과학자 토론 검토 요약

2026-09-30. 사용자 요청대로 ARIS scientist와 scientific-skills scientist가 독립 입장을 만든 뒤 직접 반론·수정·공동 권고를 주고받았다. PI가 원문·수식·범위·실험 수를 점검하고 통합 계획을 갱신했다. 같은 모델 계열의 협업이며 외부 실증 검증이나 신규성 인증이 아니다.

## 논의와 반영

| 단계 | 실제 기록 | 핵심 내용 |
|---|---|---|
| 독립 검토 | aris/01_position.md, scientific/01_position.md | 각각 세 논문·기존 자료를 읽고 가정/확률 공간/관측량 및 최소 대조 제안 |
| 직접 교차 논의 | 각02_response.md | primary ensemble 우선순위, hard-flat 반례와soft 잔여, correlated prior 검사의 우선순위, full-curve 비용에 실제 이견과 수정 |
| 공동 권고 | aris/03_joint_recommendation.md | E_opt primary 유지, E_joint/soft challenger 공개, 공유reference와238–350fit bridge, 조건부 대형실험 |
| 최종 동의 | scientific/03_joint_response.md | 합의 확인, teacher/dataset/optimizer 계층 명료화 |
| PI 초안 재검토 | 각04_pi_draft_review.md | ARIS: 수식·source 범위, scientific: protocol·산술·branch 검사 |
| 마지막 수정 | PI 통합 문서 | Rotation의a 고정/encoder 변환 조건, basic17 gate, hardmask 정의, M2→W3 명칭 정리 |

## 하나의 결론

VG-SAE의 원리적 설명과 밀도 추정은 한 논문으로 유지한다. Phase1의 conditional response, model内 uncertainty와 Phase2의 training ensemble variation을 구분하고, 그 사이의 전이를 작은 paired bridge로 검증한다. 기존7k–10kfit 계획은 첫 실행 단위에서 해제하고, bridge·개발·lock 이후의 본검증으로 단계화한다.

Operational 이견은 해소됐다. 어떤 ensemble/readout이 실제로 density/recovery에 유익한지, 효과가 학습 budget에 얼마나 의존하는지는 여전히 미확인이다. 주가설 변경은 가능하되 기존 가설을 지우지 않고 새version/fresh confirmation을 요구한다.

## 검토의 한계

Nature의 출판본 전체는 접근하지 못해 상세 가정은 공개 저자 preprint를 사용했다. Replica 계산 전체를 재증명하지 않았다. 새 학습/GPU/모델 코드 변경은 없고 Phase2 결과도 없다. 9월29일READY9.27은 당시계획에만 해당하며 이번검토의 점수로재사용하지 않는다.

최종근거·PI판정: `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`. Source의 버전/읽기 깊이와 실제 토론 원문을 모두 보존한다. Scientific skills의 방법론 provenance도 해당보고서에 별도 인용했다.
