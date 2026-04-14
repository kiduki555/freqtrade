# OHIO Market State Engine — Freqtrade Migration

> **Status**: M0 Foundation 완료, M1 완료, M2 완료, M3 완료, M4 완료, M5 완료
> **이전 문서**: 이 문서는 기존 migration.md (Freqtrade-centric 접근)을 대체합니다.
> **설계 방향**: Ohio-centric, Freqtrade as Adapter


## 한 줄 요약

7축 시장 상태 벡터로 regime을 판단하고, regime에 맞는 전략 모드를 자동 선택하여
Freqtrade의 IStrategy 콜백을 통해 실행하는 모듈러 모놀리스.


## 핵심 설계 결정

### 1. Ohio-centric, Freqtrade as Adapter

기존 migration.md는 Freqtrade 안에 ohio_integration/ 레이어를 넣는 방식이었다.
새 설계는 반대다:

- `ohio/core/` — 순수 Python, Freqtrade 의존성 0. 시장 상태, 전략 라우팅, 리스크 전부 여기.
- `ohio/adapters/freqtrade/` — Freqtrade IStrategy 콜백을 ohio/core에 위임하는 얇은 껍질.
- Freqtrade는 실행 엔진(backtest/paper/live)만 담당. 우리 로직은 건드리지 않는다.

이렇게 하면:
- Freqtrade CLI(`backtesting`, `trade`, `hyperopt`)가 그대로 작동한다.
- ohio/core는 독립적으로 테스트 가능하다.
- 나중에 다른 실행 엔진(vectorbt, 자체 엔진) 연결도 adapter만 추가하면 된다.

### 2. Regime-Centric 전략 모델

개별 전략(trend_following, defensive 등)은 독립적인 전략이 아니다.
regime에 종속된 "행동 모드"다.

- 비교 단위는 개별 모드가 아닌 **파이프라인 전체 설정**이다.
- 전략 비교 = OhioConfig 세트 비교 (stabilizer params, routing weights 등).
- `--strategy-list OhioV1 OhioV2`로 Freqtrade 네이티브 비교 가능.

### 3. 레짐 전환 시 포지션 처리 — 단계적 전환

regime이 바뀌면 기존 포지션을 즉시 청산하지 않는다. 단계적으로 처리한다:

```
regime 전환 감지
  → 즉시: 새 모드의 신규 진입만 허용, 이전 모드 진입 차단
  → 단계 1: 기존 포지션의 stoploss를 타이트하게 조임 (절반)
  → 단계 2: transition_risk > 0.8 이면 즉시 exit
  → 단계 3: 수익 중이면 수익 보전 exit
  → 단계 4: 전환 후 12봉(12시간) 경과하면 강제 exit
```

이유:
- 즉시 청산은 전환 구간에서 whipsaw 손실이 크다.
- 단계적 전환은 자연스럽게 포지션을 마무리하면서 리스크를 관리한다.

### 4. Freqtrade 콜백 매핑

IStrategy의 콜백에 ohio 로직을 위임한다. OhioThinStrategy 하나가 전체를 처리한다.

| Freqtrade 콜백 | Ohio 역할 | 해당 FT 이슈 |
|---|---|---|
| `populate_indicators()` | 전체 7-Phase 파이프라인 실행 | FT-017 |
| `populate_entry_trend()` | fitness 기반 진입 시그널 + enter_tag | FT-017 |
| `populate_exit_trend()` | 레짐 급변 시 배치 exit 시그널 | FT-017 |
| `custom_stake_amount()` | Kelly 사이징 + 모드별 배수 | FT-013 |
| `leverage()` | 모드별 레버리지 | FT-013 |
| `confirm_trade_entry()` | RiskEngine 15개 규칙 게이트 | FT-014 |
| `order_filled()` | metadata 저장 (strategy_mode 등) | FT-018 |
| `custom_stoploss()` | 모드별 동적 스탑 + 레짐전환 시 타이트닝 | FT-015 |
| `custom_exit()` | 다중 exit 조건 + 레짐전환 단계적 exit | FT-015 |

주의사항:
- Freqtrade 최신 버전에서 메서드 이름은 `order_filled`이다 (`on_fill_order` 아님).
- `self.stoploss = -0.20`이 hard ceiling. `custom_stoploss()`는 이보다 넓은 스탑 불가.
- `use_custom_stoploss = True` 설정 필수 (안 하면 custom_stoploss 호출 안 됨).
- `startup_candle_count = 4320` (180일 @ 1h) — Freqtrade가 warmup 자동 처리.

### 5. Multi-Strategy 구현 방식

Freqtrade는 1 bot = 1 strategy 모델이다. 우리는 entry_tag로 모드를 전달한다:

```
populate_entry_trend():
  BTC fitness → trend_following 최고 → enter_tag = "ohio_trend_following"
  ETH fitness → defensive 최고     → enter_tag = "ohio_defensive"

custom_stake_amount(entry_tag="ohio_trend_following"):
  → trend_following 사이징 적용

order_filled():
  → trade.set_custom_data("strategy_mode", trade.enter_tag)

custom_stoploss():
  → trade.get_custom_data("strategy_mode") → 모드별 스탑
```


## 7-Phase Pipeline

```
① INPUT          OHLCV 1h + 4h (@informative)
      ↓
② FEATURE         FeatureBuilder: 18개 primitive feature 추출 (벡터화)
      ↓
③ COMPUTE         Normalizer → Factor Calculator (7축 raw state vector)
      ↓
④ PERSIST         State Stabilizer: EMA smoothing + Numba JIT (jump gate + dwell)
      ↓
⑤ MAP             Fitness Estimator → Policy Generator
      ↓
⑥ RISK/PORTFOLIO  RiskEngine(15규칙) + PositionSizer(Kelly) + DrawdownController
      ↓
⑦ EXEC            Freqtrade 콜백을 통한 주문 실행
```


## 7축 StateVector

| Axis | Range | Type | Description |
|---|---|---|---|
| trend_persistence | [-1, 1] | per-symbol | 추세 방향 및 강도 |
| volatility_level | [0, 1] | per-symbol | 정규화된 변동성 |
| downside_pressure | [0, 1] | per-symbol | 하방 리스크 강도 |
| liquidity_stress | [0, 1] | per-symbol | 유동성 악화 정도 |
| relative_strength | [0, 1] | per-symbol | peer basket 대비 강도 |
| correlation_stress | [0, 1] | shared | 자산 간 상관관계 |
| breadth_dispersion | [0, 1] | shared | 수익률 분산 정도 |


## State Stabilizer 설계

3단 파이프라인. 상태 전환의 persistence를 보장한다.

```
raw state_vector
  → EMA Smoothing (pandas ewm, 벡터화)
  → Jump Penalty Gate + Dwell Controller (Numba JIT, 통합 단일 함수)
  → stabilized state_vector
```

핵심:
- Jump Gate와 Dwell Controller는 순차 의존성이 있다 (stabilized[t-1] 참조).
- 따라서 둘을 합쳐서 단일 Numba JIT 함수로 구현한다.
- 이중 모드: `StateStabilizer` (배치, backtest용) + `StreamingStateStabilizer` (live용).
- 두 모드의 출력은 동일 입력에 대해 bit-exact 일치해야 한다 (FT-020).

기본값: `ema_alpha=0.15`, `jump_threshold=0.10`, `min_dwell=3` (M5 FT-023 A/B 테스트 기반 튜닝).


## 전략 모드별 프로파일

| Mode | 선호 조건 | 회피 조건 | Stoploss 범위 | Leverage |
|---|---|---|---|---|
| trend_following | trend high, stability high | liquidity_stress high | -0.08 ~ -0.12 (trailing) | 2x~3x |
| mean_reversion | vol moderate, trend low | trend extreme | -0.05 ~ -0.08 (fixed) | 1x~2x |
| breakout | vol expanding, rs high | corr_stress high | -0.10 ~ -0.15 + breakeven | 2x~4x |
| defensive | stability high, vol low | downside high | -0.03 ~ -0.05 (tight) | 1x |


## Drawdown Controller

포트폴리오 레벨 드로우다운 단계적 대응:

| Drawdown | 반응 |
|---|---|
| < 5% | 정상 운영 |
| 5% ~ 10% | size_multiplier × 0.5, 신규 진입 제한 |
| 10% ~ 15% | 신규 진입 차단, 기존 포지션 유지 |
| ≥ 15% | Kill Switch 발동 → 전 포지션 청산 |

Kill Switch 해제는 수동만 가능 (자동 해제 금지).


## 파일 구조

```
freqtrade/ohio/
├── __init__.py                          # __version__ = "0.1.0"
├── core/
│   ├── domain/
│   │   └── models.py                    # StateVector, StateMeta, StrategyFitness,
│   │                                    # ExecutionPolicy, MarketStateSnapshot,
│   │                                    # ExecutionIntent, DataMode, StrategyMode
│   ├── market_state/
│   │   ├── feature_builder.py           # 18개 feature 추출
│   │   ├── normalizer.py                # percentile rank 180일
│   │   ├── mtf_merger.py                # @informative 4h 병합
│   │   ├── factors/                     # 7축 각각의 계산 모듈
│   │   │   ├── trend.py
│   │   │   ├── volatility.py
│   │   │   ├── downside.py
│   │   │   ├── liquidity.py
│   │   │   ├── relative_strength.py
│   │   │   ├── correlation.py           # shared
│   │   │   └── breadth.py               # shared
│   │   ├── stabilizer.py                # Numba JIT (batch + streaming)
│   │   └── meta_calculator.py           # transition_risk, confidence, stability
│   ├── strategy_router/
│   │   ├── fitness_estimator.py         # state × profile → score
│   │   └── policy_generator.py          # score → ExecutionPolicy
│   └── portfolio_risk/
│       ├── drawdown_controller.py       # 4단계 DD 대응
│       └── kill_switch.py               # 비상 정지
├── adapters/
│   └── freqtrade/
│       ├── thin_strategy.py             # OhioThinStrategy(IStrategy)
│       ├── position_adapter.py          # Kelly → custom_stake_amount + leverage
│       ├── risk_gate_adapter.py         # 15규칙 → confirm_trade_entry
│       ├── exit_adapter.py              # → custom_stoploss + custom_exit
│       ├── metadata_handler.py          # → order_filled
│       └── cross_asset_provider.py      # peer basket @informative
└── config/
    ├── defaults.py                      # OhioConfig frozen dataclass
    └── strategies/                      # YAML strategy profiles
        ├── trend_following.yaml
        ├── mean_reversion.yaml
        ├── breakout.yaml
        └── defensive.yaml
```


## 기존 코드 재사용

| Ohio 모듈 | 재사용율 | 대상 |
|---|---|---|
| regime-module | 100% | Factor 계산 로직, 축 정의 |
| PositionSizer (Kelly) | 100% | Kelly fraction 계산 |
| RiskEngine (15규칙) | 100% | 규칙 로직 전체 |
| ExitEngine | 85% | trailing stop, time exit, regime exit |

총 ~45K / 60K lines (75%) 재사용.


## CLI 사용법

```bash
# 백테스트
freqtrade backtesting --strategy OhioThinStrategy --config config.json

# 설정 비교 (V1 vs V2)
freqtrade backtesting --strategy-list OhioV1 OhioV2

# 하이퍼옵트
freqtrade hyperopt --strategy OhioThinStrategy --hyperopt-loss SharpeHyperOptLoss

# 페이퍼 트레이딩
freqtrade trade --strategy OhioThinStrategy --config config_dryrun.json

# 라이브
freqtrade trade --strategy OhioThinStrategy --config config_live.json

# 데이터 다운로드 (180일 warmup)
freqtrade download-data --timeframes 1h 4h --days 200
```

전략 비교 시 OhioV1, OhioV2는 OhioThinStrategy를 상속하고 `ohio_config`만 바꾼 래퍼:

```python
# user_data/strategies/OhioV2.py
from ohio.adapters.freqtrade.thin_strategy import OhioThinStrategy
from ohio.config.defaults import OhioConfig

class OhioV2(OhioThinStrategy):
    ohio_config = OhioConfig(ema_alpha=0.08, jump_threshold=0.20, min_dwell=12)
```


## 마일스톤 & Linear 이슈

| Milestone | 이슈 | 상태 |
|---|---|---|
| **M0: Foundation** | FT-001 스캐폴딩, FT-002 도메인 객체, FT-003 스켈레톤 | **Done** |
| **M1: Market State Engine** | FT-004~009 (Feature, Normalize, MTF, Factor, Stabilizer, Meta) | **Done** |
| **M2: Strategy Router** | FT-010~012 (Profile YAML, Fitness, Policy) | **Done** |
| **M3: Risk/Portfolio Adapter** | FT-013~016 (Sizer, Risk, Exit, DD/Kill) | **Done** |
| **M4: Freqtrade Integration** | FT-017~021 (Strategy 완성, metadata, cross-asset, parity, data) | **Done** |
| **M5: Validation** | FT-022~025 (Full backtest, Stabilizer A/B, Regime analysis, Paper) | **Done** |

전체 25개 이슈, Linear 프로젝트 "Freqtrade Migration"에 등록 완료.
의존성 그래프(blockedBy)도 설정 완료.


## Safety Gate 기준

| Level | 검증 | 자동/수동 |
|---|---|---|
| L1 | ruff + mypy | 자동 |
| L2 | pytest (단위/통합) | 자동 |
| L3 | Evaluator agent 코드 리뷰 | 자동 |
| L4 | backtest equity 비교 | 자동 + 수동 리뷰 |
| L5 | 수동 승인 (risk/FSM/live 관련) | 수동 |


## 주요 리스크 & 완화책

| 리스크 | 완화책 |
|---|---|
| Stabilizer batch/streaming 불일치 → backtest↔live 괴리 | FT-020: bit-exact parity 테스트 |
| Freqtrade API 변경 | ohio/core에 FT 의존성 0 유지, adapter만 수정 |
| 레짐 전환 whipsaw 손실 | 단계적 전환 (즉시 청산 금지) |
| custom_stoploss 무시됨 | use_custom_stoploss=True 필수 (Evaluator 체크) |
| metadata 손실 | order_filled()에서 저장, DB persist 검증 |
| 과적합 | FT-023: walk-forward validation (3개월 train / 1개월 test × 4) |
