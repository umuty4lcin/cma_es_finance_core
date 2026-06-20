import { useState } from 'react'
import type { AnalyzeRequest } from '../types'

interface Props {
  symbols: string[]
  loading: boolean
  onAnalyze: (req: AnalyzeRequest) => void
}

export default function Controls({ symbols, loading, onAnalyze }: Props) {
  const [symbol, setSymbol] = useState('FROTO')
  const [modelType, setModelType] = useState<'2class' | '3class'>('2class')
  const [autoOpt, setAutoOpt] = useState(true)
  const [mode, setMode] = useState('fixed')

  // 2-sinifli manuel parametreler
  const [threshold, setThreshold] = useState(50)
  // 3-sinifli manuel parametreler
  const [longThr, setLongThr] = useState(50)
  const [shortThr, setShortThr] = useState(50)
  // Ortak
  const [stopLoss, setStopLoss] = useState(2)
  const [takeProfit, setTakeProfit] = useState(4)

  const submit = () => {
    const req: AnalyzeRequest = {
      symbol, model_type: modelType, auto_optimize: autoOpt, mode,
      stop_loss: stopLoss / 100, take_profit: takeProfit / 100,
    }
    if (modelType === '2class') {
      req.threshold = threshold / 100
    } else {
      req.long_threshold = longThr / 100
      req.short_threshold = shortThr / 100
    }
    onAnalyze(req)
  }

  return (
    <div className="panel controls">
      <h3 style={{ marginTop: 0 }}>Simulasyon Ayarlari</h3>

      <label className="ctrl-label">Hisse Senedi</label>
      <select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="ctrl-input">
        {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>

      <label className="ctrl-label">Model Tipi</label>
      <select value={modelType} onChange={(e) => setModelType(e.target.value as '2class' | '3class')}
              className="ctrl-input">
        <option value="2class">2-Sinifli (sadece long)</option>
        <option value="3class">3-Sinifli (long + short)</option>
      </select>

      <label className="ctrl-label">Cuzdan Modu</label>
      <select value={mode} onChange={(e) => setMode(e.target.value)} className="ctrl-input">
        <option value="fixed">Sabit Kasa (10.000 TL)</option>
        <option value="compound">Dinamik Cuzdan (Bilesik)</option>
      </select>

      <label className="ctrl-check">
        <input type="checkbox" checked={autoOpt} onChange={(e) => setAutoOpt(e.target.checked)} />
        Otomatik Optimize Et (CMA-ES)
      </label>

      {!autoOpt && modelType === '2class' && (
        <div className="manual-params">
          <label className="ctrl-label">Esik: %{threshold}</label>
          <input type="range" min={45} max={90} step={0.5} value={threshold}
                 onChange={(e) => setThreshold(+e.target.value)} className="ctrl-range" />
        </div>
      )}

      {!autoOpt && modelType === '3class' && (
        <div className="manual-params">
          <label className="ctrl-label">Long Esik: %{longThr}</label>
          <input type="range" min={35} max={90} step={0.5} value={longThr}
                 onChange={(e) => setLongThr(+e.target.value)} className="ctrl-range" />
          <label className="ctrl-label">Short Esik: %{shortThr}</label>
          <input type="range" min={35} max={90} step={0.5} value={shortThr}
                 onChange={(e) => setShortThr(+e.target.value)} className="ctrl-range" />
        </div>
      )}

      {!autoOpt && (
        <div className="manual-params">
          <label className="ctrl-label">Stop-Loss: %{stopLoss}</label>
          <input type="range" min={0.5} max={15} step={0.1} value={stopLoss}
                 onChange={(e) => setStopLoss(+e.target.value)} className="ctrl-range" />
          <label className="ctrl-label">Take-Profit: %{takeProfit}</label>
          <input type="range" min={1} max={30} step={0.1} value={takeProfit}
                 onChange={(e) => setTakeProfit(+e.target.value)} className="ctrl-range" />
        </div>
      )}

      <button onClick={submit} disabled={loading} className="run-btn">
        {loading ? 'Analiz ediliyor...' : 'Simulasyonu Baslat'}
      </button>
      {autoOpt && <p className="hint">
        CMA-ES optimizasyonu birkac saniye surebilir
        {modelType === '3class' ? ' (4-boyutlu arama).' : '.'}
      </p>}
    </div>
  )
}
