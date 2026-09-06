import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Upload, Sliders, Brain, Layers, BarChart2, FileText,
  Loader2, CheckCircle2, AlertCircle, ChevronRight,
  Clock, Activity, Zap,
} from 'lucide-react'

export const STAGES = [
  { id: 'upload',     label: 'Upload Received',         short: 'Upload',       icon: Upload,    approx: 0.14 },
  { id: 'preprocess', label: 'Preprocessing',           short: 'Preprocess',   icon: Sliders,   approx: 0.18 },
  { id: 'inference',  label: 'Model Inference',         short: 'Inference',    icon: Brain,     approx: 0.26 },
  { id: 'segment',    label: 'Segmentation Post-Proc',  short: 'Segmentation', icon: Layers,    approx: 0.12 },
  { id: 'quant',      label: 'Quantitative Analysis',   short: 'Quantitative', icon: BarChart2, approx: 0.10 },
  { id: 'report',     label: 'Report Generation',       short: 'Report',       icon: FileText,  approx: 0.20 },
]

function fmtElapsed(ms) {
  if (ms == null || ms < 0) ms = 0
  const totalSec = Math.floor(ms / 1000)
  const m  = Math.floor(totalSec / 60)
  const s  = totalSec % 60
  const cs = Math.floor((ms % 1000) / 10)
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(cs).padStart(2,'0')}`
}

/* ── Circular progress ring ────────────────────────────────────────────── */
function CircularProgress({ value, size = 140, stroke = 9, active, indeterminate }) {
  const radius       = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const clamped      = Math.max(0, Math.min(100, value ?? 0))
  const offset       = circumference - (clamped / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {active && (
        <div className="absolute inset-0 rounded-full ring-2 ring-medical-300/40
                        shadow-[0_0_30px_rgba(21,110,245,0.18)] transition-all duration-500" />
      )}
      <svg width={size} height={size} className="-rotate-90">
        {/* Track */}
        <circle cx={size/2} cy={size/2} r={radius}
                fill="none" stroke="#e2e8f0" strokeWidth={stroke} />
        {/* Arc */}
        <circle cx={size/2} cy={size/2} r={radius}
                fill="none"
                stroke="url(#gradRingLight)"
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={offset}
                className="transition-[stroke-dashoffset] duration-300 ease-out" />
        <defs>
          <linearGradient id="gradRingLight" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#22d3ee" />
            <stop offset="50%"  stopColor="#156ef5" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
      </svg>
      {indeterminate && active && (
        <div className="absolute inset-0 rounded-full border-2 border-medical-300/30 animate-ping" />
      )}
      {/* Centre label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`font-mono font-bold text-2xl tracking-tight
          ${clamped >= 100 ? 'text-emerald-600' : 'text-slate-800'}`}>
          {Math.round(clamped)}
          <span className="text-sm text-slate-400 ml-0.5">%</span>
        </span>
        <span className={`text-[10px] uppercase tracking-[0.2em] font-semibold mt-0.5
          ${clamped >= 100 ? 'text-emerald-500' : 'text-slate-400'}`}>
          {clamped >= 100 ? 'Complete' : active ? 'Processing' : 'Idle'}
        </span>
      </div>
    </div>
  )
}

/* ── Stage row ─────────────────────────────────────────────────────────── */
function StageRow({ stage, idx, status, detail }) {
  const Icon     = stage.icon
  const isDone   = status === 'done'
  const isActive = status === 'active'
  const isError  = status === 'error'

  const iconCls = isDone
    ? 'bg-emerald-50 border-emerald-300 text-emerald-600'
    : isActive
    ? 'bg-medical-50 border-medical-300 text-medical-600 shadow-md shadow-medical-100'
    : isError
    ? 'bg-red-50 border-red-300 text-red-500'
    : 'bg-slate-100 border-slate-200 text-slate-300'

  const labelCls = isDone
    ? 'text-emerald-700'
    : isActive
    ? 'text-slate-900 font-semibold'
    : isError
    ? 'text-red-600'
    : 'text-slate-400'

  const detailCls = isDone
    ? 'text-slate-400'
    : isActive
    ? 'text-slate-600'
    : isError
    ? 'text-red-500'
    : 'text-slate-300'

  const connectorCls = isDone ? 'bg-emerald-300' : 'bg-slate-200'

  return (
    <div className="flex items-start gap-4 group">
      <div className="flex flex-col items-center shrink-0">
        <div className={`w-11 h-11 rounded-2xl border flex items-center justify-center
                         transition-all duration-300 ${iconCls}`}>
          {isDone ? (
            <CheckCircle2 className="w-5 h-5" />
          ) : isError ? (
            <AlertCircle className="w-5 h-5" />
          ) : isActive ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <span className="text-sm font-bold">{idx + 1}</span>
          )}
        </div>
        {idx < STAGES.length - 1 && (
          <div className="relative w-px h-7 my-1 overflow-hidden">
            <div className={`absolute inset-0 transition-colors duration-300 ${connectorCls}`} />
            {isActive && (
              <div className="absolute inset-0 bg-gradient-to-b from-medical-400/80
                              via-medical-300/30 to-transparent
                              animate-[pulse_1.2s_ease-in-out_infinite]" />
            )}
          </div>
        )}
      </div>

      <div className="flex-1 pb-5 last:pb-0 pt-1">
        <div className="flex items-center gap-2 flex-wrap">
          <Icon className={`w-4 h-4 transition-colors duration-300 ${
            isDone ? 'text-emerald-500' : isActive ? 'text-medical-500'
            : isError ? 'text-red-500' : 'text-slate-300'
          }`} />
          <p className={`text-sm tracking-tight transition-colors duration-300 ${labelCls}`}>
            {stage.label}
          </p>
          {isActive && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                             bg-medical-50 border border-medical-200
                             text-[10px] font-semibold text-medical-600 uppercase tracking-wider">
              <span className="w-1 h-1 rounded-full bg-medical-500 animate-pulse" />
              Running
            </span>
          )}
          {isDone && (
            <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-[0.2em]">
              OK
            </span>
          )}
        </div>
        <p className={`text-[12px] mt-1 leading-relaxed ${detailCls}`}>
          {detail
            || (isActive && 'Executing stage…')
            || (isDone   && 'Stage complete.')
            || (isError  && 'Stage failed.')
            || 'Queued.'}
        </p>
      </div>
    </div>
  )
}

/* ── Main component ────────────────────────────────────────────────────── */
export default function AnalysisPipeline({
  stageStates, errorMsg, reportError,
  onRetry, onBack, onRetryReport,
  overallProgress, startTime,
}) {
  const [elapsed, setElapsed] = useState(0)
  const rafRef   = useRef(null)
  const startRef = useRef(startTime || Date.now())

  useEffect(() => {
    startRef.current = startTime || Date.now()
  }, [startTime])

  useEffect(() => {
    const tick = () => {
      setElapsed(Date.now() - startRef.current)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  const { activeStage, doneCount, isErrored, isDone, hasActive } = useMemo(() => {
    let activeId = null, firstErrorId = null, done = 0
    for (const s of STAGES) {
      const st = stageStates?.[s.id] || 'pending'
      if (st === 'active'  && activeId      == null) activeId      = s.id
      if (st === 'error'   && firstErrorId  == null) firstErrorId  = s.id
      if (st === 'done') done++
    }
    const allDone = done === STAGES.length
    const errored = firstErrorId != null
    const active  = STAGES.find(s => s.id === (
      activeId || firstErrorId || (allDone ? STAGES[STAGES.length - 1].id : null)
    ))
    return { activeStage: active, doneCount: done,
             isErrored: errored, isDone: allDone, hasActive: activeId != null }
  }, [stageStates])

  const ringValue = overallProgress ?? (doneCount / STAGES.length) * 100

  const firstErrorStageIdx = useMemo(() => {
    if (!stageStates) return -1
    return STAGES.findIndex(s => stageStates[s.id] === 'error')
  }, [stageStates])

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Main pipeline card ─────────────────────────────────────── */}
      <div className="glass-card overflow-hidden">
        <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 p-5 sm:p-6">

          {/* LEFT: stages */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-5">
              <div className="p-1.5 rounded-xl bg-medical-50 border border-medical-200">
                <Brain className="w-4 h-4 text-medical-600" />
              </div>
              <div>
                <h2 className="section-title mb-0">Analysis Pipeline</h2>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Attention U-Net Tumour Segmentation Workstation
                </p>
              </div>
            </div>

            <div className="pl-1 pr-1">
              {STAGES.map((s, i) => (
                <StageRow
                  key={s.id}
                  stage={s}
                  idx={i}
                  status={stageStates?.[s.id] || 'pending'}
                  detail={
                    stageStates?.[s.id] === 'error' && i === firstErrorStageIdx
                      ? errorMsg
                      : undefined
                  }
                />
              ))}
            </div>
          </div>

          {/* RIGHT: HUD */}
          <div className="lg:w-[260px] shrink-0
                          lg:border-l lg:border-slate-200 lg:pl-6
                          border-t lg:border-t-0 border-slate-200 pt-5 lg:pt-0">
            <div className="flex flex-col items-center">
              <CircularProgress
                value={ringValue}
                active={hasActive && !isErrored}
                indeterminate={hasActive && ringValue > 0 && ringValue < 100}
              />

              {/* Stage badge */}
              <div className="mt-5 w-full">
                <div className={`rounded-2xl border p-4 transition-all duration-300 ${
                  isDone
                    ? 'bg-emerald-50 border-emerald-200'
                    : isErrored
                    ? 'bg-red-50 border-red-200'
                    : 'bg-medical-50 border-medical-200'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-slate-400">
                      {isDone ? 'Pipeline' : isErrored ? 'Failed at' : 'Current Stage'}
                    </span>
                    {isDone   ? <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    : isErrored ? <AlertCircle   className="w-4 h-4 text-red-500" />
                    : hasActive ? <Zap           className="w-4 h-4 text-medical-500 animate-pulse" />
                    :             <Clock         className="w-4 h-4 text-slate-400" />}
                  </div>
                  <p className={`text-sm font-bold leading-tight ${
                    isDone ? 'text-emerald-700' : isErrored ? 'text-red-600' : 'text-slate-800'
                  }`}>
                    {activeStage?.label || 'Initialising…'}
                  </p>
                  {!isDone && !isErrored && activeStage && (
                    <div className="mt-2 flex items-center gap-1">
                      {STAGES.map(s => (
                        <div key={s.id}
                             className={`flex-1 h-1 rounded-full transition-all duration-300 ${
                               STAGES.indexOf(activeStage) > STAGES.indexOf(s)
                                 ? 'bg-emerald-400'
                                 : s.id === activeStage.id
                                 ? 'bg-medical-500'
                                 : 'bg-slate-200'
                             }`} />
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Counter chips */}
              <div className="mt-4 grid grid-cols-2 gap-2.5 w-full">
                <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <Clock className="w-3 h-3" />
                    <span className="text-[9px] uppercase tracking-[0.18em] font-bold">Elapsed</span>
                  </div>
                  <p className="font-mono font-bold text-[15px] text-slate-800 tabular-nums">
                    {fmtElapsed(elapsed)}
                  </p>
                </div>
                <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                  <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                    <Activity className="w-3 h-3" />
                    <span className="text-[9px] uppercase tracking-[0.18em] font-bold">Stages</span>
                  </div>
                  <p className="font-mono font-bold text-[15px] text-slate-800 tabular-nums">
                    {doneCount}<span className="text-slate-400">/{STAGES.length}</span>
                  </p>
                </div>
              </div>

              {/* Linear progress */}
              <div className="mt-4 w-full">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[9px] uppercase tracking-[0.18em] font-bold text-slate-400">
                    Total Progress
                  </span>
                  <span className="text-[11px] font-mono font-semibold text-slate-600 tabular-nums">
                    {Math.round(ringValue)}%
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden relative">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-medical-500
                               to-emerald-500 transition-all duration-500 ease-out"
                    style={{ width: `${ringValue}%` }}
                  />
                  {hasActive && ringValue > 0 && ringValue < 100 && (
                    <div className="absolute inset-y-0 w-16 bg-gradient-to-r
                                    from-transparent via-white/60 to-transparent
                                    animate-[shimmer_1.5s_ease-in-out_infinite]"
                         style={{ left: `${ringValue}%`, transform: 'translateX(-100%)' }} />
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Report error ─────────────────────────────────────────────── */}
      {reportError && !errorMsg && (
        <div className="flex items-start gap-3 p-5 rounded-2xl bg-red-50 border border-red-200">
          <div className="p-2 rounded-xl bg-red-100 border border-red-200 flex-shrink-0">
            <AlertCircle className="w-5 h-5 text-red-500" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-bold text-red-700">Report generation failed</p>
            <p className="text-xs text-red-600 mt-1 break-words font-mono">{reportError}</p>
            <p className="text-[11px] text-slate-500 mt-2">
              Segmentation results are preserved. You may retry report generation without
              re-running the analysis pipeline.
            </p>
          </div>
          {onRetryReport && (
            <button onClick={onRetryReport} className="btn-primary flex items-center gap-2 text-sm shrink-0">
              <Loader2 className="w-4 h-4" />
              Retry Report
            </button>
          )}
        </div>
      )}

      {/* ── Pipeline error ───────────────────────────────────────────── */}
      {errorMsg && !reportError && (
        <div className="flex items-start gap-3 p-5 rounded-2xl bg-red-50 border border-red-200">
          <div className="p-2 rounded-xl bg-red-100 border border-red-200 flex-shrink-0">
            <AlertCircle className="w-5 h-5 text-red-500" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-bold text-red-700">Analysis failed</p>
            <p className="text-xs text-red-600 mt-1 break-words font-mono">{errorMsg}</p>
          </div>
        </div>
      )}

      {/* ── Action bar ───────────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <button onClick={onBack} className="btn-secondary flex items-center gap-2">
          <ChevronRight className="w-4 h-4 -rotate-180" />
          Back to Upload
        </button>
        {errorMsg && (
          <button onClick={onRetry} className="btn-primary flex items-center gap-2">
            <Loader2 className="w-4 h-4" />
            Retry Analysis
          </button>
        )}
      </div>
    </div>
  )
}
