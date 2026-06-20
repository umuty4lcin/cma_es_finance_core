// API yanit tipleri (api/engine.py analyze_symbol ciktisina karsilik gelir)

export interface Candle {
  time: number // unix saniye
  open: number
  high: number
  low: number
  close: number
}

export interface TimeValue {
  time: number
  value: number
}

export interface Marker {
  time: number
  price: number
}

export interface AnalyzeResponse {
  symbol: string
  model_type?: string  // '3class' ise asagidaki params 3-sinifli yapida olur
  params: {
    // 2-sinifli alanlar (model_type === '3class' iken yok)
    threshold?: number
    // 3-sinifli alanlar (model_type === '3class' iken var)
    long_threshold?: number
    short_threshold?: number
    // Ortak
    stop_loss: number
    take_profit: number
    mode: string
    auto_optimize: boolean
  }
  metrics: {
    net_profit: number
    final_equity: number
    win_rate: number
    max_drawdown: number
    calmar: number
    total_trades: number
    long_count: number
    short_count: number
  }
  candles: Candle[]
  volume: TimeValue[]
  kalman: TimeValue[]
  equity: TimeValue[]
  drawdown: TimeValue[]
  markers: {
    long: Marker[]
    short: Marker[]
    exit: Marker[]
  }
  // 2-sinifli model: tek olasilik dizisi
  predictions?: number[]
  // 3-sinifli model: ucu birden
  predictions_3class?: {
    p_down: number[]
    p_flat: number[]
    p_up: number[]
  }
  radar: {
    labels: string[]
    values: number[]
  }
  initial_capital: number
}

export interface PortfolioRequest {
  symbols: string[]
  max_positions: number
  auto_optimize: boolean
  sizing: string
}

export interface PortfolioResponse {
  summary: {
    net_profit: number
    final_equity: number
    return_pct: number
    max_drawdown: number
    win_rate: number
    calmar: number
    total_trades: number
    max_concurrent: number
  }
  comparison: {
    isolated_capital: number
    isolated_profit: number
    isolated_return_pct: number
    portfolio_capital: number
    portfolio_profit: number
    portfolio_return_pct: number
  }
  equity: TimeValue[]
  drawdown: TimeValue[]
  per_symbol: { symbol: string; pnl: number }[]
  blotter: {
    symbol: string
    exit_ts: number
    dir: number
    net_return_pct: number
    pnl: number
    bars_held: number
  }[]
  params: { symbol: string; threshold: number; stop_loss: number; take_profit: number }[]
  max_positions: number
  initial_capital: number
}

export interface AnalyzeRequest {
  symbol: string
  model_type: '2class' | '3class'
  auto_optimize: boolean
  // 2-sinifli
  threshold?: number
  // 3-sinifli
  long_threshold?: number
  short_threshold?: number
  // Ortak
  stop_loss?: number
  take_profit?: number
  mode: string
}
