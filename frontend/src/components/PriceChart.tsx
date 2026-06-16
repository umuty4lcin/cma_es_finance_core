import { useEffect, useRef } from 'react'
import {
  createChart, CandlestickSeries, HistogramSeries, LineSeries,
  createSeriesMarkers, ColorType,
} from 'lightweight-charts'
import type { AnalyzeResponse } from '../types'

// Candlestick + hacim + Kalman + alim/short isaretleri (TradingView tarzi)
export default function PriceChart({ data }: { data: AnalyzeResponse }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0e1117' },
        textColor: '#d1d4dc',
      },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#2a2f3a' },
      rightPriceScale: { borderColor: '#2a2f3a' },
      width: ref.current.clientWidth,
      height: 460,
    })

    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    candle.setData(data.candles as never)

    const kalman = chart.addSeries(LineSeries, {
      color: '#ffa600', lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
    })
    kalman.setData(data.kalman as never)

    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' }, priceScaleId: '',
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    volume.setData(data.volume.map((v) => ({ time: v.time, value: v.value, color: '#2a3f5f' })) as never)

    const markers = [
      ...data.markers.long.map((m) => ({
        time: m.time, position: 'belowBar', color: '#00e5ff', shape: 'arrowUp', text: 'AL',
      })),
      ...data.markers.short.map((m) => ({
        time: m.time, position: 'aboveBar', color: '#e040fb', shape: 'arrowDown', text: 'SHORT',
      })),
    ].sort((a, b) => (a.time as number) - (b.time as number))
    createSeriesMarkers(candle, markers as never)

    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth })
    })
    ro.observe(ref.current)

    return () => {
      ro.disconnect()
      chart.remove()
    }
  }, [data])

  return <div ref={ref} style={{ width: '100%' }} />
}
