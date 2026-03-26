from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="polymethemoney", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    timezone: str = Field(default="Asia/Seoul", alias="TIMEZONE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+asyncpg://polymethemoney:polymethemoney@postgres:5432/polymethemoney",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")

    starting_capital_usd: float = Field(default=400.0, alias="STARTING_CAPITAL_USD")
    max_positions: int = Field(default=40, alias="MAX_POSITIONS")
    max_position_usd: float = Field(default=10.0, alias="MAX_POSITION_USD")
    min_position_usd: float = Field(default=2.0, alias="MIN_POSITION_USD")
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION")
    fixed_position_usd: float = Field(default=0.0, alias="FIXED_POSITION_USD")
    daily_loss_limit_pct: float = Field(default=0.08, alias="DAILY_LOSS_LIMIT_PCT")
    weekly_loss_limit_pct: float = Field(default=0.20, alias="WEEKLY_LOSS_LIMIT_PCT")
    daily_profit_target_pct: float = Field(default=0.10, alias="DAILY_PROFIT_TARGET_PCT")
    position_stop_loss_pct: float = Field(default=0.25, alias="POSITION_STOP_LOSS_PCT")
    position_take_profit_pct: float = Field(default=0.20, alias="POSITION_TAKE_PROFIT_PCT")

    # Paper venue config (demo fills only)
    paper_exchange_venues: str = Field(default="binance,bybit", alias="PAPER_EXCHANGE_VENUES")
    paper_ticker_symbols: str = Field(
        default=(
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT"
        ),
        alias="PAPER_TICKER_SYMBOLS",
    )
    paper_market_symbol_map: str = Field(default="{}", alias="PAPER_MARKET_SYMBOL_MAP")
    paper_ticker_ttl_seconds: int = Field(default=30, alias="PAPER_TICKER_TTL_SECONDS")
    paper_ticker_spread_bps: float = Field(default=5.0, alias="PAPER_TICKER_SPREAD_BPS")
    paper_ticker_timeout_seconds: float = Field(default=2.0, alias="PAPER_TICKER_TIMEOUT_SECONDS")
    paper_fallback_to_polymarket: bool = Field(default=False, alias="PAPER_FALLBACK_TO_POLYMARKET")
    paper_binance_bookticker_url: str = Field(
        default="https://api.binance.com/api/v3/ticker/24hr",
        alias="PAPER_BINANCE_BOOKTICKER_URL",
    )
    paper_bybit_tickers_url: str = Field(
        default="https://api.bybit.com/v5/market/tickers",
        alias="PAPER_BYBIT_TICKERS_URL",
    )

    auto_threshold: int = Field(default=90, alias="AUTO_THRESHOLD")
    semi_threshold: int = Field(default=75, alias="SEMI_THRESHOLD")
    order_ttl_seconds: int = Field(default=300, alias="ORDER_TTL_SECONDS")
    limit_requote_retries: int = Field(default=3, alias="LIMIT_REQUOTE_RETRIES")
    limit_requote_step: float = Field(default=0.005, alias="LIMIT_REQUOTE_STEP")
    taker_fee_bps: float = Field(default=10.0, alias="TAKER_FEE_BPS")
    slippage_bps: float = Field(default=15.0, alias="SLIPPAGE_BPS")

    min_open_interest_usd: float = Field(default=50000.0, alias="MIN_OPEN_INTEREST_USD")
    max_spread: float = Field(default=0.025, alias="MAX_SPREAD")
    min_hourly_volume_usd: float = Field(default=10000.0, alias="MIN_HOURLY_VOLUME_USD")
    max_open_per_market_side: int = Field(default=1, alias="MAX_OPEN_PER_MARKET_SIDE")
    signal_reentry_cooldown_seconds: int = Field(default=600, alias="SIGNAL_REENTRY_COOLDOWN_SECONDS")
    signal_min_contract_price: float = Field(default=0.05, alias="SIGNAL_MIN_CONTRACT_PRICE")
    signal_min_net_ev: float = Field(default=0.0015, alias="SIGNAL_MIN_NET_EV")
    signal_tail_prob_floor: float = Field(default=0.03, alias="SIGNAL_TAIL_PROB_FLOOR")
    signal_tail_prob_ceiling: float = Field(default=0.97, alias="SIGNAL_TAIL_PROB_CEILING")
    signal_tail_extra_net_ev: float = Field(default=0.006, alias="SIGNAL_TAIL_EXTRA_NET_EV")
    signal_side_balance_enabled: bool = Field(default=False, alias="SIGNAL_SIDE_BALANCE_ENABLED")
    signal_side_balance_window: int = Field(default=300, alias="SIGNAL_SIDE_BALANCE_WINDOW")
    signal_max_side_ratio: float = Field(default=0.70, alias="SIGNAL_MAX_SIDE_RATIO")
    signal_dominant_side_penalty: float = Field(default=0.35, alias="SIGNAL_DOMINANT_SIDE_PENALTY")
    signal_dominant_side_hard_block: bool = Field(default=False, alias="SIGNAL_DOMINANT_SIDE_HARD_BLOCK")
    signal_side_policy: str = Field(default="YES_PRIORITY", alias="SIGNAL_SIDE_POLICY")
    signal_no_min_net_ev: float = Field(default=0.0025, alias="SIGNAL_NO_MIN_NET_EV")
    signal_no_min_confidence: float = Field(default=0.62, alias="SIGNAL_NO_MIN_CONFIDENCE")
    signal_no_regime_trend_bias: float = Field(default=-0.15, alias="SIGNAL_NO_REGIME_TREND_BIAS")
    signal_no_regime_imbalance: float = Field(default=-0.10, alias="SIGNAL_NO_REGIME_IMBALANCE")
    signal_no_max_ratio: float = Field(default=0.30, alias="SIGNAL_NO_MAX_RATIO")
    auto_tune_zero_fill_enabled: bool = Field(default=True, alias="AUTO_TUNE_ZERO_FILL_ENABLED")
    auto_tune_window_minutes: int = Field(default=60, alias="AUTO_TUNE_WINDOW_MINUTES")
    auto_tune_min_signals: int = Field(default=5, alias="AUTO_TUNE_MIN_SIGNALS")
    auto_tune_min_position_usd: float = Field(default=1.0, alias="AUTO_TUNE_MIN_POSITION_USD")
    auto_tune_apply_once: bool = Field(default=True, alias="AUTO_TUNE_APPLY_ONCE")
    paper_perf_tune_enabled: bool = Field(default=True, alias="PAPER_PERF_TUNE_ENABLED")
    paper_perf_tune_interval_hours: int = Field(default=3, alias="PAPER_PERF_TUNE_INTERVAL_HOURS")
    paper_perf_tune_interval_hours_secondary: int = Field(default=12, alias="PAPER_PERF_TUNE_INTERVAL_HOURS_SECONDARY")
    paper_perf_tune_min_pnl_usd: float = Field(default=0.0, alias="PAPER_PERF_TUNE_MIN_PNL_USD")
    paper_perf_tune_min_trades: int = Field(default=1, alias="PAPER_PERF_TUNE_MIN_TRADES")
    paper_perf_tune_target: str = Field(default="both", alias="PAPER_PERF_TUNE_TARGET")
    paper_perf_tune_only_in_paper: bool = Field(default=True, alias="PAPER_PERF_TUNE_ONLY_IN_PAPER")

    train_days: int = Field(default=180, alias="TRAIN_DAYS")
    valid_days: int = Field(default=30, alias="VALID_DAYS")
    auto_build_training_from_db: bool = Field(default=True, alias="AUTO_BUILD_TRAINING_FROM_DB")
    training_lookback_days: int = Field(default=180, alias="TRAINING_LOOKBACK_DAYS")
    training_horizon_minutes: int = Field(default=5, alias="TRAINING_HORIZON_MINUTES")
    training_min_move: float = Field(default=0.0005, alias="TRAINING_MIN_MOVE")
    trend_recency_half_life_days: float = Field(default=7.0, alias="TREND_RECENCY_HALF_LIFE_DAYS")
    trend_intraday_retrain_minutes: int = Field(default=30, alias="TREND_INTRADAY_RETRAIN_MINUTES")
    trend_adaptive_enabled: bool = Field(default=True, alias="TREND_ADAPTIVE_ENABLED")
    trend_window_size: int = Field(default=600, alias="TREND_WINDOW_SIZE")
    trend_align_boost: float = Field(default=0.5, alias="TREND_ALIGN_BOOST")
    trend_revert_penalty: float = Field(default=0.7, alias="TREND_REVERT_PENALTY")
    micro_revert_enabled: bool = Field(default=True, alias="MICRO_REVERT_ENABLED")
    micro_revert_imb_threshold: float = Field(default=0.10, alias="MICRO_REVERT_IMB_THRESHOLD")
    micro_revert_max_vol: float = Field(default=0.04, alias="MICRO_REVERT_MAX_VOL")
    micro_revert_boost: float = Field(default=0.20, alias="MICRO_REVERT_BOOST")
    model_min_rows: int = Field(default=400, alias="MODEL_MIN_ROWS")
    incremental_retrain_hour_utc: int = Field(default=2, alias="INCREMENTAL_RETRAIN_HOUR_UTC")
    full_retrain_day_of_week: str = Field(default="sun", alias="FULL_RETRAIN_DAY_OF_WEEK")
    full_retrain_hour_utc: int = Field(default=3, alias="FULL_RETRAIN_HOUR_UTC")

    arb_enabled: bool = Field(default=True, alias="ARB_ENABLED")
    arb_min_net_edge_pct: float = Field(default=0.01, alias="ARB_MIN_NET_EDGE_PCT")
    arb_max_legs: int = Field(default=20, alias="ARB_MAX_LEGS")
    arb_exec_window_ms: int = Field(default=1500, alias="ARB_EXEC_WINDOW_MS")
    arb_early_exit_edge_capture: float = Field(default=0.5, alias="ARB_EARLY_EXIT_EDGE_CAPTURE")
    arb_capital_share: float = Field(default=0.5, alias="ARB_CAPITAL_SHARE")
    arb_fee_buffer_bps: float = Field(default=10.0, alias="ARB_FEE_BUFFER_BPS")
    arb_slippage_buffer_bps: float = Field(default=15.0, alias="ARB_SLIPPAGE_BUFFER_BPS")
    arb_scan_interval_seconds: int = Field(default=45, alias="ARB_SCAN_INTERVAL_SECONDS")
    arb_market_limit: int = Field(default=300, alias="ARB_MARKET_LIMIT")

    gate_paper_days: int = Field(default=3, alias="GATE_PAPER_DAYS")
    gate_paper_min_pf: float = Field(default=1.10, alias="GATE_PAPER_MIN_PF")
    gate_paper_max_dd_pct: float = Field(default=0.04, alias="GATE_PAPER_MAX_DD_PCT")
    gate_paper_min_trades: int = Field(default=10, alias="GATE_PAPER_MIN_TRADES")
    gate_paper_max_violations: int = Field(default=0, alias="GATE_PAPER_MAX_VIOLATIONS")
    gate_hist_window_days: int = Field(default=60, alias="GATE_HIST_WINDOW_DAYS")
    gate_hist_min_pf: float = Field(default=1.05, alias="GATE_HIST_MIN_PF")
    gate_hist_max_dd_pct: float = Field(default=0.08, alias="GATE_HIST_MAX_DD_PCT")
    gate_hist_max_ece: float = Field(default=0.05, alias="GATE_HIST_MAX_ECE")

    polymarket_gamma_url: str = Field(
        default="https://gamma-api.polymarket.com/markets",
        alias="POLYMARKET_GAMMA_URL",
    )
    polymarket_data_api_url: str = Field(
        default="https://data-api.polymarket.com",
        alias="POLYMARKET_DATA_API_URL",
    )
    polymarket_clob_rest_url: str = Field(
        default="https://clob.polymarket.com",
        alias="POLYMARKET_CLOB_REST_URL",
    )
    polymarket_use_bulk_books: bool = Field(default=False, alias="POLYMARKET_USE_BULK_BOOKS")
    polymarket_subgraph_url: str = Field(
        default="https://api.thegraph.com/subgraphs/name/polymarket/matic-markets",
        alias="POLYMARKET_SUBGRAPH_URL",
    )
    polymarket_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        alias="POLYMARKET_WS_URL",
    )
    polymarket_api_key: str = Field(default="", alias="POLYMARKET_API_KEY")
    polymarket_api_secret: str = Field(default="", alias="POLYMARKET_API_SECRET")
    polymarket_api_passphrase: str = Field(default="", alias="POLYMARKET_API_PASSPHRASE")

    trading_mode: str = Field(default="paper", alias="TRADING_MODE")
    demo_unlimited: bool = Field(default=False, alias="DEMO_UNLIMITED")
    demo_paper_hardlock: bool = Field(default=False, alias="DEMO_PAPER_HARDLOCK")
    demo_reset_paper_on_startup: bool = Field(default=False, alias="DEMO_RESET_PAPER_ON_STARTUP")
    allow_live_without_gate: bool = Field(default=False, alias="ALLOW_LIVE_WITHOUT_GATE")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    report_interval_minutes: int = Field(default=60, alias="REPORT_INTERVAL_MINUTES")

    training_intraday_file: str = Field(default="data/training_intraday.csv", alias="TRAINING_INTRADAY_FILE")
    training_settlement_file: str = Field(default="data/training_settlement.csv", alias="TRAINING_SETTLEMENT_FILE")


def get_settings() -> Settings:
    return Settings()
