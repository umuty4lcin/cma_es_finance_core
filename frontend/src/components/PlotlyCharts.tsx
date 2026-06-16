import { useEffect, useRef } from 'react'
import * as PlotlyModule from 'plotly.js-dist-min'
import type { AnalyzeResponse } from '../types'

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

// --- Kasa + Drawdown (iki panel) ---
export function EquityDrawdownChart({ data }: { data: AnalyzeResponse }) {
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
export function ProbHistogram({ data }: { data: AnalyzeResponse }) {
  const traces: any[] = [{
    x: data.predictions, type: 'histogram', nbinsx: 40,
    marker: { color: '#636efa' },
  }]
  const layout: any = {
    ...DARK, height: 300, margin: { t: 30, r: 20, b: 40, l: 50 },
    xaxis: { title: 'Yukselis Olasiligi', gridcolor: '#1c2230', range: [0, 1] },
    yaxis: { title: 'Mum Sayisi', gridcolor: '#1c2230' },
    shapes: [{
      type: 'line', x0: data.params.threshold, x1: data.params.threshold,
      y0: 0, y1: 1, yref: 'paper', line: { color: '#ef553b', dash: 'dash', width: 2 },
    }],
    annotations: [{
      x: data.params.threshold, y: 1, yref: 'paper', text: `Esik %${(data.params.threshold * 100).toFixed(0)}`,
      showarrow: false, font: { color: '#ef553b', size: 11 }, xanchor: 'left',
    }],
  }
  return <Plot data={traces} layout={layout} config={CONFIG} style={{ width: '100%' }} useResizeHandler />
}
