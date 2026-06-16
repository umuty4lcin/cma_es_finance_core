import { useState } from 'react'
import type { AnalyzeRequest } from '../types'

interface Props {
  symbols: string[]
  loading: boolean
  onAnalyze: (req: AnalyzeRequest) => void
}

export default function Controls({ symbols, loading, onAnalyze }: Props) {
  const [symbol, setSymbol] = useState('FROTO')
  const [autoOpt, setAutoOpt] = useState(true)
  const [mode, setMode] = useState('fixed')
  const [allowShort, setAllowShort] = useState(false)
  const [threshold, setThreshold] = useState(50)
  const [stopLoss, setStopLoss] = useState(2)
  const [takeProfit, setTakeProfit] = useState(4)

  const submit = () => {
    onAnalyze({
      symbol, auto_optimize: autoOpt, mode, allow_short: allowShort,
      threshold: threshold / 100, stop_loss: stopLoss / 100, take_profit: takeProfit / 100,
    })
  }

  return (
    <div className="panel controls">
      <h3 style={{ marginTop: 0 }}>Simulasyon Ayarlari</h3>

      <label className="ctrl-label">Hisse Senedi</label>
      <select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="ctrl-input">
        {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>

      <label className="ctrl-label">Cuzdan Modu</label>
      <select value={mode} onChange={(e) => setMode(e.target.value)} className="ctrl-input">
        <option value="fixed">Sabit Kasa (10.000 TL)</option>
        <option value="compound">Dinamik Cuzdan (Bilesik)</option>
      </select>

      <label className="ctrl-check">
        <input type="checkbox" checked={allowShort} onChange={(e) => setAllowShort(e.target.checked)} />
        Short Islem (deneysel)
      </label>

      <label className="ctrl-check">
        <input type="checkbox" checked={autoOpt} onChange={(e) => setAutoOpt(e.target.checked)} />
        Otomatik Optimize Et (CMA-ES)
      </label>

      {!autoOpt && (
        <div className="manual-params">
          <label className="ctrl-label">Esik: %{threshold}</label>
          <input type="range" min={45} max={90} step={0.5} value={threshold}
                 onChange={(e) => setThreshold(+e.target.value)} className="ctrl-range" />
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
      {autoOpt && <p className="hint">CMA-ES optimizasyonu birkac saniye surebilir.</p>}
    </div>
  )
}
