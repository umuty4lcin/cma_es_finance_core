import { useState } from 'react'
import { runPortfolio } from '../api'
import type { PortfolioResponse } from '../types'
import { EquityDrawdownChart, PerSymbolPnLBar } from './PlotlyCharts'

const ALL = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']

function dirLabel(d: number) {
  return d === 1 ? 'LONG' : d === -1 ? 'SHORT' : '-'
}

export default function PortfolioPage() {
  const [maxPos, setMaxPos] = useState(4)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<PortfolioResponse | null>(null)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      const data = await runPortfolio({
        symbols: ALL, max_positions: maxPos, auto_optimize: true, sizing: 'fixed',
      })
      setRes(data)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="panel" style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
        <div>
          <div className="ctrl-label" style={{ margin: 0 }}>Es zamanli max pozisyon: <b>{maxPos}</b></div>
          <input type="range" min={1} max={8} value={maxPos}
                 onChange={(e) => setMaxPos(+e.target.value)} style={{ width: 220 }} />
        </div>
        <button onClick={run} disabled={loading} className="run-btn" style={{ width: 240, marginTop: 0 }}>
          {loading ? 'Hesaplaniyor...' : 'Portfoy Simulasyonu Calistir'}
        </button>
        <span className="hint">10 hisse, paylasimli 10.000 TL havuz. CMA-ES + backtest birkac dk surebilir.</span>
      </div>

      <p className="hint" style={{ marginBottom: 16 }}>
        Not: Bu modul tez kapsami disidir; gelecek calisma olarak sunulan paylasimli-sermaye portfoy demosudur.
      </p>

      {error && <div className="error-box">Hata: {error}</div>}
      {loading && <div className="empty-state">Portfoy hesaplaniyor, lutfen bekleyin...</div>}

      {res && !loading && (
        <>
          <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
            <div className="kpi-card">
              <div className="kpi-label">Net Kar (TL)</div>
              <div className="kpi-value" style={{ color: res.summary.net_profit >= 0 ? '#00cc96' : '#ef553b' }}>
                {res.summary.net_profit.toLocaleString('tr-TR')}
              </div>
              <div className="kpi-sub">%{res.summary.return_pct} getiri</div>
            </div>
            <div className="kpi-card"><div className="kpi-label">Max Drawdown</div>
              <div className="kpi-value" style={{ color: '#ef553b' }}>%{res.summary.max_drawdown}</div></div>
            <div className="kpi-card"><div className="kpi-label">Calmar</div>
              <div className="kpi-value" style={{ color: '#ffa600' }}>{res.summary.calmar}</div></div>
            <div className="kpi-card"><div className="kpi-label">Kazanma Orani</div>
              <div className="kpi-value">%{res.summary.win_rate}</div></div>
            <div className="kpi-card"><div className="kpi-label">Max Es Zamanli</div>
              <div className="kpi-value">{res.summary.max_concurrent} / {res.max_positions}</div></div>
          </div>

          <div className="panel">
            <h3 className="panel-title">Portfoy Kasa Buyumesi ve Risk</h3>
            <EquityDrawdownChart data={res} />
          </div>

          <div className="grid-2">
            <div className="panel">
              <h3 className="panel-title">Hisse Bazinda PnL Katkisi</h3>
              <PerSymbolPnLBar data={res.per_symbol} />
            </div>
            <div className="panel">
              <h3 className="panel-title">Sermaye Verimliligi</h3>
              <table className="cmp-table">
                <thead><tr><th>Senaryo</th><th>Sermaye</th><th>Net Kar</th><th>Getiri %</th></tr></thead>
                <tbody>
                  <tr>
                    <td>Izole (her hisse 10k)</td>
                    <td>{res.comparison.isolated_capital.toLocaleString('tr-TR')} TL</td>
                    <td>{res.comparison.isolated_profit.toLocaleString('tr-TR')} TL</td>
                    <td>%{res.comparison.isolated_return_pct}</td>
                  </tr>
                  <tr style={{ color: '#00cc96', fontWeight: 600 }}>
                    <td>Portfoy (tek 10k, max {res.max_positions})</td>
                    <td>{res.comparison.portfolio_capital.toLocaleString('tr-TR')} TL</td>
                    <td>{res.comparison.portfolio_profit.toLocaleString('tr-TR')} TL</td>
                    <td>%{res.comparison.portfolio_return_pct}</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">Mutlak kar yaniltici (izole 10x sermaye). Adil olcut: getiri %.</p>
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title">Trade Blotter ({res.blotter.length} islem)</h3>
            <div className="blotter-wrap">
              <table className="blotter">
                <thead>
                  <tr><th>#</th><th>Hisse</th><th>Yon</th><th>Cikis</th><th>Getiri %</th><th>PnL (TL)</th><th>Bar</th></tr>
                </thead>
                <tbody>
                  {res.blotter.map((t, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{t.symbol}</td>
                      <td>{dirLabel(t.dir)}</td>
                      <td>{new Date(t.exit_ts * 1000).toLocaleString('tr-TR')}</td>
                      <td style={{ color: t.net_return_pct >= 0 ? '#00cc96' : '#ef553b' }}>
                        %{t.net_return_pct.toFixed(2)}
                      </td>
                      <td style={{ color: t.pnl >= 0 ? '#00cc96' : '#ef553b' }}>{t.pnl.toFixed(2)}</td>
                      <td>{t.bars_held}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!res && !loading && (
        <div className="empty-state">"Portfoy Simulasyonu Calistir"a basin.</div>
      )}
    </div>
  )
}
