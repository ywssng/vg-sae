# 계획 문서 검증 — 2026-09-29

- Canonical proposal/plan/tracker/paper/brief/contract와 run 사본 및 timestamp 사본의 byte 일치 확인.
- JSON 상태 파일 parse, 로컬 Markdown 링크 존재, code fence 짝 확인.
- Immutable Problem Anchor가 round0와 최종제안에서 동일함을 확인.
- 7개 절의 내부 분량 합10쪽 확인.
- Core7240/10376 fits, 연장포함7264/10400 equivalents, mechanism subset world IDs 및 제안비용25.22–144.44GPUh 산술 확인.
- Pooled support uncertainty의 heterogeneity 식을 유리수로 만든 under/over-selection 예시에서 직접 합과 대조. 이는 문서의 대수 검산이며 Phase2 성능실험이 아니다.
- 기존 Phase1 run과 원래 LaTeX 원고에 diff가 없음을 확인.
- `git diff --check` 통과. 문서 변경이므로 Python model test suite나 GPU 학습은 실행하지 않음.
- 실제3차례 secondary reviewer 결과를 보존. Final READY9.27은 계획준비도에 대한 same-family provisional 판단이다.

Phase2 metric schema·estimator·runner는 다음 구현 작업이다. 이 검증을 구현 완료, cutoff calibration, density/recovery 성능 또는 신규성 입증으로 해석하지 않는다.
