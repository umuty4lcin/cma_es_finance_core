import type { AnalyzeRequest, AnalyzeResponse } from './types'

export async function fetchSymbols(): Promise<string[]> {
  const r = await fetch('/api/symbols')
  if (!r.ok) throw new Error('Sembol listesi alinamadi')
  const d = await r.json()
  return d.symbols
}

export async function analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const r = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: 'Bilinmeyen hata' }))
    throw new Error(err.detail ?? 'Analiz basarisiz')
  }
  return r.json()
}
