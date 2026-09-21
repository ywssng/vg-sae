# P2: beta 初기화/추정 경로 결과

현재 VG 방법 안에서 learned beta 초기값1, train-energy moment 초기화,
minibatch-profiled reference를36개 모델로 비교했다. 모든cal 결과를 먼저 모아
조건별recipe를동결한뒤각predefinedarm의test를평가했다. early1001/final6000updates.

**일관된 초기화 개선은 확인하지 못했다. 기본값을 바꿀 근거가 없다.**

| amplitude | gamma | recipe | test hard latent error | F1 | hard L0 | hard EV |
|---|---:|---|---:|---:|---:|---:|
| exponential | 2 | learned | 1.12136 | 0.21238 | 6.056 | 0.9416 |
| exponential | 2 | learned_moment | 1.13603 | 0.20502 | 5.891 | 0.9377 |
| exponential | 2 | profiled | 1.12042 | 0.21190 | 5.824 | 0.9344 |
| exponential | 6 | learned | 0.86399 | 0.18811 | 0.727 | 0.5679 |
| exponential | 6 | learned_moment | 0.85416 | 0.19335 | 0.737 | 0.5755 |
| exponential | 6 | profiled | 0.84177 | 0.19598 | 0.732 | 0.5711 |
| constant | 2 | learned | 1.20330 | 0.18336 | 9.342 | 0.9571 |
| constant | 2 | learned_moment | 1.19576 | 0.18005 | 9.038 | 0.9530 |
| constant | 2 | profiled | 1.18177 | 0.19640 | 9.216 | 0.9575 |
| constant | 6 | learned | 1.01713 | 0.03489 | 0.123 | 0.1035 |
| constant | 6 | learned_moment | 1.01698 | 0.03493 | 0.122 | 0.1039 |
| constant | 6 | profiled | 1.01860 | 0.03225 | 0.120 | 0.1027 |

표는세 paired seed의평균이며서로다른gamma의sparsity를같다고보지않는다.
Exponential/gamma6의moment초기화는평균error를약.010낮추지만한seed에서는악화되고,
gamma2에서는모든seed가악화했다. Constant/gamma6는거의차이가없다.
Trainstationary beta 대비 learned beta비율은final약.994–1.000이었다.
이는 beta의학습이이setting에서진행됐다는뜻이며전체encoder/dictionary수렴을증명하지않는다.

고정gamma에서profiledreference와learned의절대loss는직접비교하지않는다.
Profiling은batch별log energy를사용하여globallearnedbeta와다른stochasticobjective다.
본실험은warmup효과나장기Stage2의undertraining,모든VG의quality를판정하지않는다.
일부조건의latenterror>1및낮은F1은이작은고밀도setting의약한recovery를그대로드러낸다.
이를숨기거나test로gamma/recipe를다시고르지않았다.
