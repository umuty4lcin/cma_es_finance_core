import { useEffect, useState } from 'react'
import { fetchSymbols, analyze } from './api'
import type { AnalyzeRequest, AnalyzeResponse } from './types'
import Controls from './components/Controls'
import KpiCards from './components/KpiCards'
import PriceChart from './components/PriceChart'
import { EquityDrawdownChart, RadarChart, ProbHistogram } from './components/PlotlyCharts'
import PortfolioPage from './components/PortfolioPage'
import './App.css'

function App() {
  const [tab, setTab] = useState<'single' | 'portfolio'>('single')
  const [symbols, setSymbols] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)

  useEffect(() => {
    fetchSymbols().then(setSymbols).catch((e) => setError(e.message))
  }, [])

  const onAnalyze = async (req: AnalyzeRequest) => {
    setLoading(true)
    setError(null)
    try {
      const data = await analyze(req)
      setResult(data)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="topbar">
        <div className="brand">
          <h1>CMA-ES Finance Manager</h1>
          <span className="subtitle">Global LSTM + CMA-ES Evrimsel Risk Optimizasyonu | BIST 15dk</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'single' ? 'tab active' : 'tab'} onClick={() => setTab('single')}>
            Tek Hisse Analizi
          </button>
          <button className={tab === 'portfolio' ? 'tab active' : 'tab'} onClick={() => setTab('portfolio')}>
            Portfoy
          </button>
        </nav>
      </div>

      {tab === 'portfolio' ? (
        <div className="portfolio-wrap"><PortfolioPage /></div>
      ) : (
      <div className="app-layout">
        <aside className="sidebar">
          <Controls symbols={symbols} loading={loading} onAnalyze={onAnalyze} />
        </aside>

        <main className="main">
        {error && <div className="error-box">Hata: {error}</div>}

        {!result && !loading && (
          <div className="empty-state">
            Soldan bir hisse secip "Simulasyonu Baslat"a basin.
          </div>
        )}

        {loading && <div className="empty-state">Analiz ediliyor, lutfen bekleyin...</div>}

        {result && !loading && (
          <>
            <div className="params-bar">
              <b>{result.symbol}</b>
              {result.model_type === '3class' ? (
                <>
                  &nbsp;|&nbsp; Long Esik: %{((result.params.long_threshold ?? 0) * 100).toFixed(2)}
                  &nbsp;|&nbsp; Short Esik: %{((result.params.short_threshold ?? 0) * 100).toFixed(2)}
                </>
              ) : (
                <> &nbsp;|&nbsp; Esik: %{((result.params.threshold ?? 0) * 100).toFixed(2)}</>
              )}
              &nbsp;|&nbsp; SL: %{(result.params.stop_loss * 100).toFixed(2)}
              &nbsp;|&nbsp; TP: %{(result.params.take_profit * 100).toFixed(2)}
              &nbsp;|&nbsp; Mod: {result.params.mode}
              {result.model_type === '3class' && <span className="badge" style={{ background: 'rgba(168,85,247,0.15)', color: '#c084fc' }}>3-sinifli (long+short)</span>}
              {result.params.auto_optimize && <span className="badge">CMA-ES optimize</span>}
            </div>

            <KpiCards data={result} />

            <div className="panel">
              <h3 className="panel-title">{result.symbol} — Fiyat, Kalman ve Sinyaller</h3>
              <PriceChart data={result} />
            </div>

            <div className="panel">
              <h3 className="panel-title">Kasa Buyumesi ve Risk (Drawdown)</h3>
              <EquityDrawdownChart data={result} />
            </div>

            <div className="grid-2">
              <div className="panel">
                <h3 className="panel-title">Hisse Durumu Radari</h3>
                <RadarChart data={result} />
              </div>
              <div className="panel">
                <h3 className="panel-title">LSTM Olasilik Dagilimi</h3>
                <ProbHistogram data={result} />
              </div>
            </div>
          </>
        )}
        </main>
      </div>
      )}
    </div>
  )
}

export default App
