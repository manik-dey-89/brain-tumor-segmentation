import React, { useEffect, useState } from 'react'
import { Clock, RefreshCw, Trash2, ChevronRight, BarChart2, AlertTriangle } from 'lucide-react'
import { getHistory, getMetrics, clearHistory } from '../utils/api'
import { fmtDate, fmtPct, fmtMetric, metricColour, confidenceColour } from '../utils/helpers'
import toast from 'react-hot-toast'

function StatCard({ label, value, colour = 'text-medical-600' }) {
  return (
    <div className="metric-badge">
      <span className="label-text mb-2">{label}</span>
      <span className={`text-xl font-bold font-mono ${colour}`}>{value ?? '—'}</span>
    </div>
  )
}

function HistoryRow({ record }) {
  const [expanded, setExpanded] = useState(false)
  const tumour   = record.tumor_detected
  const study    = record.study || {}
  const studyId  = record.study_id || study.study_id
  const patient  = record.patient_name || study.full_name
  const modality = record.modality || study.mri_modality || study.modality
  const scanDate = record.scan_date || study.scan_date

  return (
    <div className={`bg-white rounded-xl overflow-hidden border transition-all shadow-sm ${
      tumour ? 'border-red-200' : 'border-emerald-200'
    }`}>
      <button
        className="w-full flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4
                   px-5 py-4 text-left hover:bg-slate-50 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-4 flex-1 min-w-0">
          <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
            tumour ? 'bg-red-400' : 'bg-emerald-400'
          }`} />
          <div className="min-w-0 flex-1 text-left">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm font-mono text-medical-600 tracking-wider truncate">
                {studyId || '—'}
              </span>
            </div>
            <p className="text-xs text-slate-400 truncate mt-0.5">
              <span className="text-slate-600">{patient || 'Unknown patient'}</span>
              {modality && <> · <span className="text-slate-500">{modality}</span></>}
              {scanDate && <> · <span className="text-slate-400">{String(scanDate)}</span></>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-4 ml-6 sm:ml-0">
          <span className="text-xs text-slate-400 truncate max-w-[120px] hidden md:block">
            {record.filename || '—'}
          </span>
          <span className={`text-sm font-mono flex-shrink-0 ${
            tumour ? 'text-red-500' : 'text-emerald-600'
          }`}>
            {record.tumor_percentage != null
              ? `${Number(record.tumor_percentage).toFixed(2)}%`
              : '—'}
          </span>
          {record.confidence != null && (
            <span className={`text-xs px-2.5 py-1 rounded-full border flex-shrink-0
                              ${confidenceColour(record.confidence)}`}>
              {(record.confidence * 100).toFixed(0)}%
            </span>
          )}
          <span className="text-xs text-slate-400 flex-shrink-0 hidden lg:block w-[140px] text-right">
            {fmtDate(record.timestamp)}
          </span>
          <ChevronRight className={`w-4 h-4 text-slate-300 flex-shrink-0 transition-transform ${
            expanded ? 'rotate-90' : ''
          }`} />
        </div>
      </button>

      {expanded && (
        <div className="px-5 pb-5 pt-1 border-t border-slate-100 animate-fade-in">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
            {record.metrics && Object.entries(record.metrics)
              .filter(([, v]) => v != null)
              .map(([k, v]) => (
                <div key={k} className="metric-badge py-3">
                  <span className="label-text mb-1">{k}</span>
                  <span className={`text-base font-bold font-mono ${metricColour(v)}`}>
                    {Number(v).toFixed(4)}
                  </span>
                </div>
              ))
            }
          </div>
          {record.prediction_id && (
            <p className="text-xs text-slate-400 mt-3 font-mono">
              ID: {record.prediction_id}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default function HistoryPage() {
  const [history, setHistory] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [page,    setPage]    = useState(0)
  const PAGE_SIZE = 20

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [hRes, mRes] = await Promise.all([
        getHistory(PAGE_SIZE, page * PAGE_SIZE),
        getMetrics(),
      ])
      setHistory(hRes.data)
      setMetrics(mRes.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [page])

  const handleClear = async () => {
    if (!confirm('Clear all prediction history?')) return
    try {
      await clearHistory()
      toast.success('History cleared')
      load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-8">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-900">
            <Clock className="w-8 h-8 text-medical-600" />
            Prediction History
          </h1>
          <p className="text-slate-500 mt-1">
            Recent segmentation runs ({history?.total ?? '…'} total)
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={load}        className="btn-secondary p-2.5">
            <RefreshCw className="w-4 h-4" />
          </button>
          <button onClick={handleClear} className="btn-secondary p-2.5 hover:border-red-300 hover:text-red-500">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Aggregate metrics */}
      {metrics && (
        <div className="glass-card">
          <h2 className="section-title flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-medical-600" />
            Session Statistics
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Total Analyses"   value={metrics.total}                colour="text-slate-800" />
            <StatCard label="Tumour Detected"  value={metrics.tumor_detected_count} colour="text-red-500" />
            <StatCard
              label="Detection Rate"
              value={metrics.tumor_detection_rate != null
                ? `${(metrics.tumor_detection_rate * 100).toFixed(1)}%` : '—'}
              colour="text-amber-600"
            />
            <StatCard
              label="Avg Confidence"
              value={metrics.avg_confidence != null
                ? `${(metrics.avg_confidence * 100).toFixed(1)}%` : '—'}
              colour="text-medical-600"
            />
          </div>
          {metrics.avg_dice != null && (
            <p className="text-xs text-slate-400 mt-3">
              Average Dice Score (when GT provided): {metrics.avg_dice?.toFixed(4)}
            </p>
          )}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="glass-card flex items-center justify-center py-12">
          <RefreshCw className="w-6 h-6 text-medical-500 animate-spin" />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-red-50 border border-red-200">
          <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Records */}
      {!loading && history && (
        <>
          {history.records.length === 0 ? (
            <div className="glass-card text-center py-16">
              <Clock className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-400">No predictions yet.</p>
              <p className="text-slate-300 text-sm mt-1">Run an analysis to see results here.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.records.map((r, i) => (
                <HistoryRow key={r.prediction_id || i} record={r} />
              ))}
            </div>
          )}

          {history.total > PAGE_SIZE && (
            <div className="flex items-center justify-center gap-4">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
                className="btn-secondary disabled:opacity-30"
              >
                Previous
              </button>
              <span className="text-sm text-slate-500">
                Page {page + 1} of {Math.ceil(history.total / PAGE_SIZE)}
              </span>
              <button
                disabled={(page + 1) * PAGE_SIZE >= history.total}
                onClick={() => setPage((p) => p + 1)}
                className="btn-secondary disabled:opacity-30"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
