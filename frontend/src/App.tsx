import { useEffect, useState } from 'react'

function App() {
  const [health, setHealth] = useState<string>('...')
  const [symbols, setSymbols] = useState<string[]>([])
  const [selected, setSelected] = useState<string>('')

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setHealth(d.status))
      .catch(() => setHealth('hata'))

    fetch('/api/symbols')
      .then((r) => r.json())
      .then((d) => {
        setSymbols(d.symbols)
        setSelected(d.symbols[0] ?? '')
      })
      .catch(() => {})
  }, [])

  return (
    <div
      style={{
        fontFamily: 'system-ui, sans-serif',
        padding: 40,
        color: '#e6e6e6',
        background: '#0e1117',
        minHeight: '100vh',
      }}
    >
      <h1 style={{ margin: 0 }}>CMA-ES Finance Manager</h1>
      <p style={{ opacity: 0.7 }}>Global LSTM + CMA-ES Evrimsel Risk Optimizasyonu | BIST</p>

      <p>
        Backend durumu:{' '}
        <b style={{ color: health === 'ok' ? '#00cc96' : '#ef553b' }}>{health}</b>
      </p>

      <div style={{ marginTop: 16 }}>
        <label style={{ marginRight: 8 }}>Hisse Senedi:</label>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          style={{ padding: '6px 12px', background: '#1a1f2b', color: '#e6e6e6', border: '1px solid #333', borderRadius: 6 }}
        >
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <p style={{ marginTop: 24, opacity: 0.5 }}>
        R0 iskelet calisiyor — {symbols.length} sembol API'den yuklendi. Secili: {selected}
      </p>
    </div>
  )
}

export default App
