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
  params: {
    threshold: number
    stop_loss: number
    take_profit: number
    mode: string
    allow_short: boolean
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
  predictions: number[]
  radar: {
    labels: string[]
    values: number[]
  }
  initial_capital: number
}

export interface AnalyzeRequest {
  symbol: string
  auto_optimize: boolean
  threshold?: number
  stop_loss?: number
  take_profit?: number
  mode: string
  allow_short: boolean
}
