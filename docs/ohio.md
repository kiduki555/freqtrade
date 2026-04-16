# OHIO Market State Engine

> Freqtrade 위에서 동작하는 7축 시장 상태 기반 자동 전략 라우팅 시스템.

---

## 개요

OHIO는 Freqtrade를 **실행 엔진**으로만 사용하고, 모든 로직은 `freqtrade/ohio/` 안에서 독립적으로 동작한다. 전략 코드가 짧은 이유는 `OhioThinStrategy`가 실제로는 얇은 어댑터(thin adapter)이기 때문이다 — 진입/청산/사이징 결정은 전부 `ohio/core/`에서 이루어진다.

### 핵심 차이 (기존 Freqtrade 전략 대비)

| 항목 | 일반 Freqtrade 전략 | OHIO |
|------|---------------------|------|
| 전략 수 | 1개 고정 | 4개 모드 자동 전환 |
| 진입 결정 | 인디케이터 규칙 | 7축 시장 상태 fitness score |
| 포지션 사이징 | 고정 stake | Kelly 기반 동적 사이징 |
| 레버리지 | 고정 | 모드별 자동 조정 |
| 스탑로스 | 고정 또는 trailing | 모드별 동적 + 레짐 전환 시 타이트닝 |
| hyperopt 파라미터 | 인디케이터 임계값 | 모드별 진입 조건 + 청산 타이밍 분리 |
| backtest | `freqtrade backtesting` | `freqtrade backtesting` + `scripts/full_backtest.py` |

---

## 아키텍처

```
OHLCV 1h
  │
  ▼
① FeatureBuilder          18개 primitive feature 추출
  │
  ▼
② Normalizer              percentile rank, 180일 rolling window
  │
  ▼
③ FactorCalculator        7축 raw state vector 계산
  │
  ▼
④ StateStabilizer         EMA smoothing + Jump Gate + Dwell (Numba JIT)
  │
  ▼
⑤ MetaCalculator          transition_risk, confidence, stability
  │
  ▼
⑥ FitnessEstimator        state vector × strategy profile → 4개 fitness score
  │
  ▼
⑦ PolicyGenerator         fitness → ExecutionPolicy (enabled, size_multiplier 등)
  │
  ▼
OhioThinStrategy          Freqtrade 콜백에 위임
```

### 7축 StateVector

| 축 | 범위 | 설명 |
|----|------|------|
| `trend_persistence` | [-1, 1] | 추세 방향 및 강도 |
| `volatility_level` | [0, 1] | 정규화된 변동성 |
| `downside_pressure` | [0, 1] | 하방 리스크 강도 |
| `liquidity_stress` | [0, 1] | 유동성 악화 정도 |
| `relative_strength` | [0, 1] | peer basket 대비 강도 |
| `correlation_stress` | [0, 1] | 자산 간 상관관계 (cross-asset) |
| `breadth_dispersion` | [0, 1] | 수익률 분산 정도 (cross-asset) |

### 4개 전략 모드

| 모드 | 선호 조건 | 스탑로스 | 레버리지 |
|------|-----------|----------|----------|
| `trend_following` | trend↑, rs↑, vol 보통 | -8% ~ -12% (trailing) | 2x ~ 3x |
| `mean_reversion` | trend 없음, vol 보통 | -5% ~ -8% (fixed) | 1x ~ 2x |
| `breakout` | vol↑, rs↑, dispersion↑ | -10% ~ -15% (breakeven) | 2x ~ 4x |
| `defensive` | vol↓, downside↓, stability↑ | -3% ~ -5% (tight) | 1x |

---

## 파일 구조

```
freqtrade/ohio/
├── core/
│   ├── domain/models.py              # 도메인 객체 (StateVector, StrategyMode 등)
│   ├── market_state/
│   │   ├── feature_builder.py        # 18개 feature 추출
│   │   ├── normalizer.py             # percentile rank 정규화
│   │   ├── factors/                  # 7축 각각의 계산 모듈
│   │   ├── stabilizer.py             # Numba JIT 상태 안정화
│   │   └── meta_calculator.py        # transition_risk, confidence
│   ├── strategy_router/
│   │   ├── fitness_estimator.py      # state × profile → score
│   │   └── policy_generator.py       # score → ExecutionPolicy
│   └── portfolio_risk/
│       ├── drawdown_controller.py    # 4단계 DD 대응
│       └── kill_switch.py            # 비상 정지
├── adapters/freqtrade/
│   ├── thin_strategy.py              # OhioThinStrategy (IStrategy 어댑터)
│   ├── entry_adapter.py              # 모드별 진입 시그널
│   ├── entries/                      # trend_following, mean_reversion, breakout, defensive
│   ├── exit_adapter.py               # custom_stoploss + custom_exit
│   ├── position_adapter.py           # Kelly 사이징 + 레버리지
│   ├── risk_gate_adapter.py          # confirm_trade_entry 15규칙 게이트
│   ├── cross_asset_provider.py       # peer basket 데이터
│   └── metadata_handler.py           # trade 메타데이터 저장
└── config/
    ├── defaults.py                   # OhioConfig frozen dataclass
    └── strategies/                   # 4개 모드 YAML 프로파일
        ├── trend_following.yaml
        ├── mean_reversion.yaml
        ├── breakout.yaml
        └── defensive.yaml

user_data/
├── config_dryrun.json                # paper trading 설정
├── config_backtest.json              # backtest 설정
└── config_download.json              # 데이터 다운로드 설정

scripts/
├── full_backtest.py                  # 파이프라인 검증 + 시그널 분석
├── regime_analysis.py                # K-means 레짐 클러스터링
├── monitor_paper.py                  # paper trading 모니터링
└── download_ohio_data.sh             # 데이터 다운로드
```

---

## 빠른 시작

### 1. 데이터 다운로드

```bash
# 180일 warmup 포함 200일치 다운로드
freqtrade download-data --config user_data/config_download.json --timeframes 1h --days 200
```

> **주의**: `startup_candle_count = 4320` (180일 @ 1h). 데이터가 부족하면 워밍업 구간이 늘어나 실제 신호가 줄어든다.

### 2. Paper Trading (Dry Run)

```bash
python -m freqtrade trade --strategy OhioThinStrategy --config user_data/config_dryrun.json
```

**FreqUI 접속**: `http://127.0.0.1:8080` (ID: `freqtrader` / PW: `freqtrader`)

### 3. 모니터링

```bash
# 별도 터미널에서 실행
python scripts/monitor_paper.py

# API URL 또는 인증 커스텀
python scripts/monitor_paper.py --api-url http://127.0.0.1:8080 --interval 60
```

알림 기준:
- Drawdown 10% 초과 시 콘솔 빨간 경고
- 24시간 이상 신규 거래 없을 시 경고
- 봇 상태가 `running`이 아닐 시 경고

로그: `user_data/logs/paper_monitor.log`

---

## Backtest

### 기본 Freqtrade Backtest

```bash
freqtrade backtesting --strategy OhioThinStrategy --config user_data/config_backtest.json --timerange 20240101-20241231
```

### OHIO 파이프라인 검증 Backtest

Freqtrade backtesting과 별개로 **파이프라인 결정성 + 시그널 분석**을 빠르게 확인할 수 있다.

```bash
# 기본 (BTC_USDT, ETH_USDT)
python scripts/full_backtest.py

# 페어 지정
python scripts/full_backtest.py --pairs BTC_USDT,ETH_USDT,ADA_USDT

# 출력 디렉토리 지정
python scripts/full_backtest.py --pairs BTC_USDT --output-dir user_data/backtest_results
```

출력 파일:
- `user_data/backtest_results/full_pipeline_report.txt` — 페어별 시그널 통계
- `user_data/backtest_results/pipeline_signals.csv` — 진입 신호 전체 CSV

> **데이터 형식**: `user_data/data/binance/BTC_USDT-1h.feather` 파일 필요.
> Freqtrade backtesting이 자동 생성하거나 `download_ohio_data.sh` 사용.

출력 예시:
```
--- BTC_USDT ---
  Total bars:        8760
  Warmup bars:       4320
  Active bars:       4440
  Entry signals:     3821
  Entry rate:        86.06%
  Avg fitness:       0.4823
  Mode distribution (entries):
    mean_reversion            52.31%
    defensive                 28.14%
    trend_following           12.40%
    breakout                   7.15%
```

### 전략 설정 비교 (A/B Test)

`OhioThinStrategy`를 상속해 `OhioConfig`만 바꾸면 된다.

```python
# user_data/strategies/OhioV2.py
from freqtrade.ohio.adapters.freqtrade.thin_strategy import OhioThinStrategy
from freqtrade.ohio.config.defaults import OhioConfig

class OhioV2(OhioThinStrategy):
    ohio_config = OhioConfig(ema_alpha=0.08, jump_threshold=0.20, min_dwell=12)
```

```bash
freqtrade backtesting --strategy-list OhioThinStrategy OhioV2 --config user_data/config_backtest.json --timerange 20240101-20241231
```

---

## Hyperopt

### 중요: OHIO Hyperopt는 5단계로 분리한다

일반 전략처럼 한 번에 전체를 최적화하면 과적합 위험이 크다. OHIO는 파이프라인 단계별로 순차 최적화한다.

| 단계 | 최적화 대상 | Space | 에폭 | 상태 |
|------|-------------|-------|------|------|
| Stage 1 | Hedge algorithm + Policy routing | `buy` | 300 | LOCKED |
| Stage 2a | Trend Following 진입 파라미터 | `buy` | 300 | LOCKED |
| Stage 2b | Mean Reversion 진입 파라미터 | `buy` | 300 | LOCKED |
| Stage 3a | Breakout 진입 파라미터 | `buy` | 200 | LOCKED |
| Stage 3b | Defensive 진입 파라미터 | `buy` | 200 | LOCKED |
| Stage 4a | Exit timing + Regime exit | `sell` | 400 | LOCKED |
| Stage 4b | Stoploss 파라미터 | `sell` | 300 | LOCKED |

현재 모든 파라미터는 `optimize=False`로 LOCKED 상태다. 재최적화 시 해당 단계 파라미터만 `optimize=True`로 변경 후 실행한다.

### 실행 방법

```bash
# Stage 2a — Trend Following 재최적화 예시
# thin_strategy.py에서 tf_* 파라미터를 optimize=True로 변경 후:
freqtrade hyperopt --strategy OhioThinStrategy --config user_data/config_backtest.json --hyperopt-loss SharpeHyperOptLoss --spaces buy --epochs 300 --timerange 20230101-20231231

# Stage 4a — Exit timing 재최적화
# time_exit_*, regime_exit_*, profit_preserve_* 를 optimize=True로 변경 후:
freqtrade hyperopt --strategy OhioThinStrategy --config user_data/config_backtest.json --hyperopt-loss SharpeHyperOptLoss --spaces sell --epochs 400 --timerange 20230101-20231231
```

### 현재 최적화된 파라미터 값

**Stage 1 — Hedge & Policy**
```
hedge_eta = 0.40
hedge_temperature = 3.6
disabled_threshold = 0.34
min_trend_confidence = 0.05
regime_cooldown_bars = 10
```

**Stage 2a — Trend Following**
```
tf_adx_threshold = 36.4
tf_hurst_threshold = 0.70
tf_kama_slope_threshold = 0.005
tf_vol_scale_target = 0.6
```

**Stage 2b — Mean Reversion**
```
mr_zscore_entry = 2.7
mr_rsi_oversold = 29
mr_rsi_overbought = 60
mr_hurst_max = 0.46
```

**Stage 3a — Breakout**
```
bo_squeeze_min_bars = 5
bo_volume_mult = 1.6
bo_donchian_window = 13
```

**Stage 3b — Defensive**
```
def_adx_max = 17.9
def_vol_scale = 0.18
def_min_trend_abs = 0.01
```

**Stage 4a — Exit Timing (Sharpe +1.43)**
```
time_exit_bars = 31
time_exit_fitness = 0.47
regime_exit_risk = 0.69
profit_preserve_profit = 0.01
profit_preserve_fitness = 0.58
```

**Stage 4b — Stoploss (Sharpe +1.47)**
```
trailing_profit_threshold = 0.048
trailing_profit_ratio = 0.52
transition_tighten_factor = 0.76
sl_hard_floor = -0.13
```

> 최적화 결과는 `user_data/strategies/OhioThinStrategy.json`에 저장된다. 이 파일이 전략 클래스 기본값을 덮어쓰므로, `minimal_roi`와 `stoploss`가 포함되어 있으면 제거해야 한다.

---

## 레짐 분석

7축 상태 벡터를 K-means로 클러스터링해 실제 시장 레짐을 분석한다.

```bash
# BTC 기본 분석 (k 자동 선택)
python scripts/regime_analysis.py --pair BTC_USDT

# k 고정
python scripts/regime_analysis.py --pair BTC_USDT --k 4

# 출력 디렉토리 지정
python scripts/regime_analysis.py --pair ETH_USDT --output-dir user_data/custom_output
```

출력 파일:
- `regime_analysis_summary.csv` — 레짐별 평균 지표 (수익률, 변동성, dominant mode 등)
- `regime_transitions.csv` — 레짐 간 전환 확률 행렬
- `regime_labeled_data.csv` — 원본 데이터에 레짐 레이블 추가

> `scikit-learn` 필요: `pip install scikit-learn`

---

## Drawdown 보호

포트폴리오 레벨 자동 대응:

| Drawdown | 동작 |
|----------|------|
| < 5% | 정상 운영 |
| 5% ~ 10% | 포지션 사이즈 × 0.5, 신규 진입 제한 |
| 10% ~ 15% | 신규 진입 전면 차단 |
| ≥ 15% | **Kill Switch** — 전 포지션 청산 |

Kill Switch는 **수동으로만 해제** 가능. 자동 해제 없음.

---

## 레짐 전환 시 포지션 처리

즉시 청산하지 않는다. 단계적으로 처리한다:

```
레짐 전환 감지
  → 즉시:     새 모드 신규 진입만 허용, 이전 모드 진입 차단
  → 단계 1:   기존 포지션 stoploss를 절반으로 타이트닝
  → 단계 2:   transition_risk > 0.8 이면 즉시 exit
  → 단계 3:   수익 중이면 수익 보전 exit
  → 단계 4:   전환 후 12봉 경과 시 강제 exit
```

---

## 주요 설정값 (OhioConfig)

`freqtrade/ohio/config/defaults.py`:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `normalization_window_days` | 180 | percentile rank 윈도우 |
| `ema_alpha` | 0.15 | 상태 안정화 EMA 계수 |
| `jump_threshold` | 0.10 | 급격한 상태 전환 억제 임계값 |
| `min_dwell` | 3 | 최소 상태 유지 봉 수 |
| `fitness_disabled_threshold` | 0.40 | 이 미만이면 진입 비활성화 |
| `max_drawdown_pct` | 0.15 | Kill Switch 발동 DD |
| `startup_candle_count` | 4320 | 워밍업 캔들 수 (180일) |

---

## 로그에서 확인할 사항

| 로그 메시지 | 의미 |
|------------|------|
| `ohio.populate_indicators \| rows=2494` | 파이프라인 정상 실행 |
| `ohio.cross_asset \| peers=9` | cross-asset 데이터 정상 수집 |
| `ohio.cross_asset \| insufficient peers (0)` | peer 데이터 없음 → 2개 축 NaN (시작 직후 일시적이면 정상) |
| `ohio.entry_blocked \| reasons=...` | RiskGate가 진입 차단 |
| `Strategy Parameter: disabled_threshold = 0.34` | JSON params 파일 로드 확인 |

---

## 자주 하는 실수

**1. minimal_roi / stoploss 가 OhioThinStrategy.json에 포함된 경우**

hyperopt 결과 JSON이 전략 값을 덮어씁니다. `user_data/strategies/OhioThinStrategy.json`에서 `minimal_roi`와 `stoploss` 항목을 제거하세요.

**2. 데이터 없이 backtesting 실행**

```bash
# 먼저 데이터 다운로드
freqtrade download-data --config user_data/config_download.json --days 200 --timeframes 1h
```

**3. startup_candle_count 부족**

데이터가 4320봉 미만이면 워밍업이 끝나지 않아 신호가 거의 없습니다. 최소 200일치 데이터를 확보하세요.

**4. cross-asset peer 경고가 계속 나오는 경우**

`informative_pairs()`가 선언되어 있으면 첫 번째 캔들 이후 자동으로 해소됩니다. 봇 시작 후 첫 사이클에서만 나오는 경고는 정상입니다.
