import type { AnalyzeResponse } from '../types'

function Card({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ color: color ?? '#e6e6e6' }}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

export default function KpiCards({ data }: { data: AnalyzeResponse }) {
  const m = data.metrics
  const ret = (m.net_profit / data.initial_capital) * 100
  return (
    <div className="kpi-grid">
      <Card label="Net Kar (TL)" value={m.net_profit.toLocaleString('tr-TR')}
            sub={`%${ret.toFixed(1)} getiri`} color={m.net_profit >= 0 ? '#00cc96' : '#ef553b'} />
      <Card label="Son Kasa" value={`${m.final_equity.toLocaleString('tr-TR')} TL`} />
      <Card label="Islem Sayisi" value={`${m.total_trades}`}
            sub={data.model_type === '3class' ? `${m.long_count} long / ${m.short_count} short` : undefined} />
      <Card label="Kazanma Orani" value={`%${m.win_rate.toFixed(1)}`} />
      <Card label="Max Drawdown" value={`%${m.max_drawdown.toFixed(1)}`} color="#ef553b" />
      <Card label="Calmar Orani" value={`${m.calmar.toFixed(1)}`} color="#ffa600" />
    </div>
  )
}
