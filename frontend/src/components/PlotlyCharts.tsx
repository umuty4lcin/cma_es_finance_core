import { useEffect, useRef } from 'react'
import * as PlotlyModule from 'plotly.js-dist-min'
import type { AnalyzeResponse, TimeValue } from '../types'

// react-plotly.js'in CJS/ESM interop sorunlarini tamamen atlatmak icin
// Plotly'yi dogrudan saran kucuk bir React bileseni.
const Plotly: any = (PlotlyModule as any).default ?? PlotlyModule

function Plot({ data, layout, config, style }: any) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el) Plotly.react(el, data, layout, config)
    return () => { if (el) Plotly.purge(el) }
  }, [data, layout, config])
  return <div ref={ref} style={style} />
}

const DARK = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#d1d4dc' },
}
const CONFIG = { displayModeBar: false, responsive: true }

// --- Kasa + Drawdown (iki panel) --- (hem tek-hisse hem portfoy yaniti kabul eder)
export function EquityDrawdownChart(
  { data }: { data: { equity: TimeValue[]; drawdown: TimeValue[]; initial_capital: number } },
) {
  const x = data.equity.map((e) => e.time * 1000)
  const eq = data.equity.map((e) => e.value)
  const ddx = data.drawdown.map((d) => d.time * 1000)
  const dd = data.drawdown.map((d) => d.value)

  const traces: any[] = [
    {
      x, y: eq, type: 'scatter', mode: 'lines', name: 'Kasa',
      line: { color: '#00cc96', width: 1.5 }, fill: 'tozeroy',
      fillcolor: 'rgba(0,204,150,0.1)', xaxis: 'x', yaxis: 'y',
    },
    {
      x: ddx, y: dd, type: 'scatter', mode: 'lines', name: 'Drawdown %',
      line: { color: '#ef553b', width: 1 }, fill: 'tozeroy',
      fillcolor: 'rgba(239,85,59,0.3)', xaxis: 'x', yaxis: 'y2',
    },
  ]
  const layout: any = {
    ...DARK, height: 380, margin: { t: 30, r: 20, b: 30, l: 60 },
    showlegend: false, hovermode: 'x unified',
    xaxis: { type: 'date', anchor: 'y2', gridcolor: '#1c2230' },
    yaxis: { domain: [0.34, 1], title: 'Kasa (TL)', gridcolor: '#1c2230' },
    yaxis2: { domain: [0, 0.24], title: 'DD %', gridcolor: '#1c2230' },
    shapes: [{
      type: 'line', x0: x[0], x1: x[x.length - 1], y0: data.initial_capital,
      y1: data.initial_capital, yref: 'y', line: { color: '#888', dash: 'dash', width: 1 },
    }],
  }
  return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} useResizeHandler />
}

// --- Portfoy: Hisse Bazinda PnL Bar ---
export function PerSymbolPnLBar({ data }: { data: { symbol: string; pnl: number }[] }) {
  const traces: any[] = [{
    x: data.map((d) => d.symbol),
    y: data.map((d) => d.pnl),
    type: 'bar',
    marker: { color: data.map((d) => (d.pnl >= 0 ? '#00cc96' : '#ef553b')) },
  }]
  const layout: any = {
    ...DARK, height: 320, margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: { gridcolor: '#1c2230' },
    yaxis: { title: 'PnL (TL)', gridcolor: '#1c2230' },
  }
  return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} />
}

// --- Hisse Durumu Radari ---
export function RadarChart({ data }: { data: AnalyzeResponse }) {
  const labels = [...data.radar.labels, data.radar.labels[0]]
  const values = [...data.radar.values, data.radar.values[0]]
  const traces: any[] = [{
    type: 'scatterpolar', r: values, theta: labels, fill: 'toself',
    line: { color: '#00cc96' }, fillcolor: 'rgba(0,204,150,0.25)', name: data.symbol,
  }]
  const layout: any = {
    ...DARK, height: 340, margin: { t: 40, r: 40, b: 30, l: 40 },
    polar: {
      bgcolor: 'rgba(0,0,0,0)',
      radialaxis: { range: [0, 100], gridcolor: '#2a2f3a', tickfont: { size: 9 } },
      angularaxis: { gridcolor: '#2a2f3a' },
    },
    showlegend: false,
  }
  return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} useResizeHandler />
}

// --- LSTM Olasilik Dagilimi ---
// 2-sinifli model: tek histogram (P_yukseliş) + esik cizgisi
// 3-sinifli model: ust uste 3 histogram (P_asagi/P_yatay/P_yukari) + 2 esik
export function ProbHistogram({ data }: { data: AnalyzeResponse }) {
  if (data.model_type === '3class' && data.predictions_3class) {
    const p = data.predictions_3class
    const traces: any[] = [
      { x: p.p_down, type: 'histogram', nbinsx: 40, name: 'P_asagi',
        marker: { color: '#ef553b' }, opacity: 0.6 },
      { x: p.p_flat, type: 'histogram', nbinsx: 40, name: 'P_yatay',
        marker: { color: '#a5a5a5' }, opacity: 0.4 },
      { x: p.p_up, type: 'histogram', nbinsx: 40, name: 'P_yukari',
        marker: { color: '#00cc96' }, opacity: 0.6 },
    ]
    const longThr = data.params.long_threshold ?? 0.5
    const shortThr = data.params.short_threshold ?? 0.5
    const layout: any = {
      ...DARK, height: 300, margin: { t: 30, r: 20, b: 40, l: 50 },
      barmode: 'overlay',
      xaxis: { title: 'Olasilik', gridcolor: '#1c2230', range: [0, 1] },
      yaxis: { title: 'Mum Sayisi', gridcolor: '#1c2230' },
      legend: { orientation: 'h', y: 1.1 },
      shapes: [
        { type: 'line', x0: longThr, x1: longThr, y0: 0, y1: 1, yref: 'paper',
          line: { color: '#00cc96', dash: 'dash', width: 2 } },
        { type: 'line', x0: shortThr, x1: shortThr, y0: 0, y1: 1, yref: 'paper',
          line: { color: '#ef553b', dash: 'dash', width: 2 } },
      ],
    }
    return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} />
  }

  // 2-sinifli yol (mevcut)
  const traces: any[] = [{
    x: data.predictions ?? [], type: 'histogram', nbinsx: 40,
    marker: { color: '#636efa' },
  }]
  const thr = data.params.threshold ?? 0.5
  const layout: any = {
    ...DARK, height: 300, margin: { t: 30, r: 20, b: 40, l: 50 },
    xaxis: { title: 'Yukselis Olasiligi', gridcolor: '#1c2230', range: [0, 1] },
    yaxis: { title: 'Mum Sayisi', gridcolor: '#1c2230' },
    shapes: [{
      type: 'line', x0: thr, x1: thr, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#ef553b', dash: 'dash', width: 2 },
    }],
    annotations: [{
      x: thr, y: 1, yref: 'paper', text: `Esik %${(thr * 100).toFixed(0)}`,
      showarrow: false, font: { color: '#ef553b', size: 11 }, xanchor: 'left',
    }],
  }
  return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} />
}
