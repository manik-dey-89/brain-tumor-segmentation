import React, { useEffect, useMemo, useState } from 'react'
import {
  User, FileImage, Activity, BarChart2, AlertTriangle,
  FileDown, Loader2, RefreshCw, Eye, Target,
  Brain, CheckCircle2, FileText, ZoomIn, Download,
  X, AlertCircle, ExternalLink, Printer, FileCheck2
} from 'lucide-react'
import MetricsPanel from './MetricsPanel'
import {
  fmtPct, fmtMs, fmtNum, fmtDate,
  confidenceColour, downloadBase64, downloadBlob
} from '../utils/helpers'
import { generateReportPdf } from '../utils/api'
import toast from 'react-hot-toast'

const NA = '—'

/* ── Small helpers ───────────────────────────────────────────────────────── */

function KV({ label, value, mono }) {
  return (
    <div className="py-2 border-b border-slate-100 last:border-0 flex justify-between gap-3 min-w-0">
      <span className="text-xs text-slate-400 uppercase tracking-wider shrink-0">{label}</span>
      <span className={`text-sm text-slate-700 truncate ${mono ? 'font-mono' : ''}`}>
        {value ?? NA}
      </span>
    </div>
  )
}

function SectionCard({ title, icon: Icon, iconTint = 'text-medical-400', children, action }) {
  return (
    <section className="glass-card">
      <div className="flex items-start justify-between gap-3 mb-4">
        <h3 className="section-title flex items-center gap-2 mb-0">
          <Icon className={`w-4 h-4 ${iconTint}`} />
          {title}
        </h3>
        {action}
      </div>
      {children}
    </section>
  )
}

function ImageTile({ src, label, download }) {
  const [zoom, setZoom] = useState(false)
  if (!src) {
    return (
      <div className="glass rounded-2xl overflow-hidden">
        <div className="aspect-square bg-slate-100 flex items-center justify-center text-slate-400 text-xs">
          Not available
        </div>
        <div className="px-3 py-2 border-t border-slate-100">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
        </div>
      </div>
    )
  }
  return (
    <div className="group bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
      <div className="relative aspect-square bg-slate-100">
        <img
          src={src}
          alt={label}
          className="w-full h-full object-contain cursor-zoom-in"
          onClick={() => setZoom(true)}
        />
        <div className="absolute top-2 right-2 flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => setZoom(true)}
            className="p-1.5 rounded-lg bg-slate-800/60 backdrop-blur-sm hover:bg-slate-800/80 transition-colors"
            title="Zoom"
          >
            <ZoomIn className="w-3.5 h-3.5 text-white" />
          </button>
          {download && (
            <button
              onClick={() => downloadBase64(src, download)}
              className="p-1.5 rounded-lg bg-slate-800/60 backdrop-blur-sm hover:bg-slate-800/80 transition-colors"
              title="Download PNG"
            >
              <Download className="w-3.5 h-3.5 text-white" />
            </button>
          )}
        </div>
      </div>
      <div className="px-3 py-2 border-t border-slate-100">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</p>
      </div>
      {zoom && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setZoom(false)}
        >
          <img src={src} alt={label} className="max-w-full max-h-full object-contain rounded-xl" />
          <p className="absolute bottom-6 text-slate-200 text-sm">Click anywhere to close</p>
        </div>
      )}
    </div>
  )
}

const LIMITATIONS = [
  'Single 2D axial-slice input — not a full multi-parametric volumetric study.',
  'Trained on synthetic MRI data — performance on clinical acquisitions may vary.',
  'Sensitivity to acquisition protocol, field strength, coil profile, and reconstruction kernel.',
  'Output must be reviewed by a qualified radiologist; not for standalone clinical use.',
  'Does not provide tumour grading, staging, or treatment recommendations.',
]

/* ── PDF Preview Modal ───────────────────────────────────────────────────── */

function PdfPreviewModal({ blob, onClose, filename }) {
  const url = useMemo(() => blob ? URL.createObjectURL(blob) : null, [blob])
  useEffect(() => {
    return () => { if (url) URL.revokeObjectURL(url) }
  }, [url])

  if (!blob || !url) return null
  return (
    <div className="fixed inset-0 z-[60] bg-slate-900/80 backdrop-blur-sm flex flex-col items-center p-3 sm:p-6 animate-fade-in">
      <div className="w-full max-w-6xl flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="p-2 rounded-xl bg-medical-50 border border-medical-200 flex-shrink-0">
            <FileText className="w-4 h-4 text-medical-600" />
          </div>
          <div className="min-w-0">
            <h3 className="text-slate-900 font-bold truncate">Report Preview</h3>
            <p className="text-xs text-slate-400 truncate font-mono">{filename}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="btn-secondary flex items-center gap-2 text-xs px-3 py-2"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Open
          </a>
          <button
            onClick={() => window.print && setTimeout(() => window.print(), 150)}
            className="btn-secondary hidden sm:flex items-center gap-2 text-xs px-3 py-2"
            title="Print / Save as PDF via system dialog"
          >
            <Printer className="w-3.5 h-3.5" />
            Print
          </button>
          <button
            onClick={() => downloadBlob(blob, filename)}
            className="btn-primary flex items-center gap-2 text-xs px-3 py-2"
          >
            <FileDown className="w-3.5 h-3.5" />
            Download
          </button>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-100 border border-slate-200 hover:bg-slate-200 transition-colors"
            aria-label="Close preview"
          >
            <X className="w-4 h-4 text-slate-600" />
          </button>
        </div>
      </div>
      <div className="flex-1 w-full max-w-6xl rounded-2xl overflow-hidden border border-slate-200 bg-white">
        <iframe
          title="Report PDF Preview"
          src={url}
          className="w-full h-full min-h-[70vh]"
        />
      </div>
      <p className="mt-3 text-xs text-slate-400 text-center max-w-2xl">
        Preview uses your browser&apos;s built-in PDF viewer. If the preview does not render,
        download the file directly using the button above.
      </p>
    </div>
  )
}

/* ── Main component ──────────────────────────────────────────────────────── */

export default function ResultDashboard({ result, onNewAnalysis, pdfBlob, pdfError, onRetryReport }) {
  // Legacy state (used if pdfBlob/pdfError props are not provided by hook)
  const [legacyGenerating, setLegacyGenerating] = useState(false)
  const [legacyBlob, setLegacyBlob] = useState(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [regenLoading, setRegenLoading] = useState(false)

  if (!result) return null

  const study = result.study || {}
  const model = result.model_info || {}
  const images = result.images || {}
  const det = result.tumor_detected

  // Prefer hook-provided pdf state; fall back to component-level state
  const hasPdfFromHook = pdfBlob != null || pdfError != null
  const activeBlob  = hasPdfFromHook ? pdfBlob  : legacyBlob
  const activeError = hasPdfFromHook ? pdfError : null
  const pdfReady = !!activeBlob

  const detBadgeCls = det
    ? 'bg-red-50 text-red-700 border-red-200'
    : 'bg-emerald-50 text-emerald-700 border-emerald-200'

  const studyId = study.study_id || result.prediction_id || 'UNKNOWN'
  const patientName = study.full_name || 'Not provided'
  const generationTs = new Date().toISOString() // roughly when report was made (fallback)
  const reportGenerationTime = result.timestamp || generationTs

  const filenameFromResult = () => {
    const sid = studyId
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 15)
    return `BrainTumorSeg_Report_${sid}_${ts}.pdf`
  }

  /* ── Actions ───────────────────────────────────────────────────────── */

  const handleGeneratePdf = async () => {
    setRegenLoading(true)
    try {
      const blob = await generateReportPdf({ prediction_result: result })
      setLegacyBlob(blob)
      toast.success('PDF report generated')
    } catch (err) {
      toast.error(err.message || 'Failed to generate PDF')
    } finally {
      setRegenLoading(false)
    }
  }

  const handleLegacyGenerate = async () => {
    setLegacyGenerating(true)
    setLegacyBlob(null)
    try {
      const blob = await generateReportPdf({ prediction_result: result })
      setLegacyBlob(blob)
      toast.success('PDF report generated')
    } catch (err) {
      toast.error(err.message || 'Failed to generate PDF')
    } finally {
      setLegacyGenerating(false)
    }
  }

  const handleDownloadPdf = () => {
    if (activeBlob) downloadBlob(activeBlob, filenameFromResult())
  }

  const handleRetryReport = async () => {
    if (onRetryReport) {
      setRegenLoading(true)
      try { await onRetryReport() } finally { setRegenLoading(false) }
      return
    }
    await handleGeneratePdf()
  }

  /* ── Render ────────────────────────────────────────────────────────── */
  return (
    <div className="space-y-5 animate-slide-up">
      {/* ══════════════════════════════════════════════════════════════════
          REPORT READY / REPORT ERROR HEADER BANNER
          ══════════════════════════════════════════════════════════════════ */}
      {pdfReady && (
        <div className="glass-card border border-emerald-200 bg-emerald-50/60 relative overflow-hidden">
          <div className="absolute -top-20 -right-20 w-72 h-72 bg-emerald-100 rounded-full blur-3xl pointer-events-none" />
          <div className="relative p-5 sm:p-6 flex flex-col lg:flex-row gap-5 items-start lg:items-center">
            <div className="flex items-start gap-4 flex-1 min-w-0">
              <div className="p-3 rounded-2xl bg-emerald-100 border border-emerald-300 flex-shrink-0">
                <div className="relative">
                  <div className="absolute inset-0 rounded-2xl bg-emerald-300/40 blur-lg animate-pulse" />
                  <FileCheck2 className="w-7 h-7 text-emerald-400 relative" />
                </div>
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap mb-1.5">
                  <h2 className="text-xl font-bold text-emerald-700 tracking-tight">
                    ✓ Report Generated
                  </h2>
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-100 border border-emerald-300 text-[10px] font-semibold text-emerald-700 uppercase tracking-wider">
                    <CheckCircle2 className="w-3 h-3" /> Ready
                  </span>
                </div>
                <p className="text-xs text-slate-500">
                  Professional radiology-style research report compiled from the completed analysis.
                </p>
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                    <span className="text-[9px] uppercase tracking-[0.18em] font-bold text-slate-400 block mb-1">
                      Study ID
                    </span>
                    <p className="font-mono text-sm text-slate-800 font-semibold truncate">{studyId}</p>
                  </div>
                  <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                    <span className="text-[9px] uppercase tracking-[0.18em] font-bold text-slate-400 block mb-1">
                      Patient Name
                    </span>
                    <p className="text-sm text-slate-800 font-semibold truncate">{patientName}</p>
                  </div>
                  <div className="rounded-xl bg-slate-50 border border-slate-200 p-3">
                    <span className="text-[9px] uppercase tracking-[0.18em] font-bold text-slate-400 block mb-1">
                      Generated
                    </span>
                    <p className="text-sm text-slate-800 font-semibold truncate">
                      {fmtDate(reportGenerationTime)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 lg:gap-2 w-full lg:w-auto">
              <button
                onClick={() => setPreviewOpen(true)}
                className="btn-secondary flex items-center justify-center gap-2 text-sm px-4"
              >
                <Eye className="w-4 h-4" />
                Preview Report
              </button>
              <button
                onClick={handleDownloadPdf}
                className="btn-primary flex items-center justify-center gap-2 text-sm px-4"
              >
                <FileDown className="w-4 h-4" />
                Download PDF Report
              </button>
            </div>
          </div>
        </div>
      )}

      {activeError && !pdfReady && (
        <div className="glass-card border border-red-200 bg-red-50/60 relative overflow-hidden animate-fade-in">
          <div className="absolute -top-20 -right-20 w-72 h-72 bg-red-100 rounded-full blur-3xl pointer-events-none" />
          <div className="relative p-5 sm:p-6 flex flex-col lg:flex-row gap-4 items-start">
            <div className="p-3 rounded-2xl bg-red-100 border border-red-300 flex-shrink-0">
              <AlertCircle className="w-7 h-7 text-red-500" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap mb-1.5">
                <h2 className="text-xl font-bold text-red-700 tracking-tight">
                  Report Generation Failed
                </h2>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-100 border border-red-300 text-[10px] font-semibold text-red-700 uppercase tracking-wider">
                  <AlertTriangle className="w-3 h-3" /> Error
                </span>
              </div>
              <p className="text-xs text-slate-500 mb-2">
                Segmentation results were computed successfully and are preserved below.
                Only the PDF report generation step encountered an error.
              </p>
              <div className="p-3 rounded-xl bg-red-50 border border-red-200 font-mono text-xs text-red-600 break-words max-h-40 overflow-y-auto">
                {activeError}
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 w-full lg:w-auto shrink-0">
              <button
                onClick={handleRetryReport}
                disabled={regenLoading}
                className="btn-primary flex items-center justify-center gap-2 text-sm px-4 disabled:opacity-50"
              >
                {regenLoading
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <RefreshCw className="w-4 h-4" />}
                {regenLoading ? 'Regenerating…' : 'Retry Report'}
              </button>
              {!hasPdfFromHook && (
                <button
                  onClick={onNewAnalysis}
                  className="btn-secondary flex items-center justify-center gap-2 text-sm px-4"
                >
                  New Analysis
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          TOP ACTION BAR (legacy controls + new analysis)
          ══════════════════════════════════════════════════════════════════ */}
      <div className="glass-card">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className={`px-3 py-2 rounded-xl border text-xs font-semibold tracking-wider uppercase ${detBadgeCls}`}>
              {det ? 'Tumour Region Detected' : 'No Tumour Region Detected'}
              <span className="ml-2 opacity-70">· AI-Assisted Research Finding</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {!hasPdfFromHook && (
              <>
                <button
                  onClick={handleLegacyGenerate}
                  disabled={legacyGenerating}
                  className="btn-secondary flex items-center gap-2 text-sm disabled:opacity-50"
                >
                  {legacyGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                  {legacyGenerating ? 'Generating PDF…' : 'Generate PDF Report'}
                </button>
                <button
                  onClick={handleDownloadPdf}
                  disabled={!activeBlob}
                  className="btn-primary flex items-center gap-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <FileDown className="w-4 h-4" />
                  Download PDF Report
                </button>
              </>
            )}
            {hasPdfFromHook && !activeBlob && (
              <button
                onClick={handleRetryReport}
                disabled={regenLoading}
                className="btn-secondary flex items-center gap-2 text-sm disabled:opacity-50"
              >
                {regenLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                {regenLoading ? 'Regenerating PDF…' : 'Regenerate PDF Report'}
              </button>
            )}
            <button
              onClick={onNewAnalysis}
              className="btn-secondary flex items-center gap-2 text-sm"
            >
              <RefreshCw className="w-4 h-4" />
              New Analysis
            </button>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 1 + 2: PATIENT SUMMARY + MRI INFO
          ══════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <SectionCard title="Patient / Study Summary" icon={User}>
          <div>
            <KV label="Study / Case ID" value={study.study_id || result.prediction_id} mono />
            <KV label="Patient ID"     value={study.patient_id} mono />
            <KV label="Full Name"      value={study.full_name} />
            <KV label="Age"            value={study.age} />
            <KV label="Gender"         value={study.gender} />
            <KV label="Scan Date"      value={study.scan_date} />
            <KV label="Modality"       value={study.mri_modality || study.modality} />
            <KV label="Referring Doctor" value={study.referring_doctor} />
            <KV label="Hospital / Clinic" value={study.hospital} />
            <KV label="Analysis Timestamp" value={fmtDate(result.timestamp)} />
          </div>
        </SectionCard>

        <SectionCard title="MRI Scan Information" icon={FileImage}>
          <div>
            <KV label="Filename"         value={result.filename} />
            <KV label="File Source Format" value={result.filename ? result.filename.split('.').pop().toUpperCase() : NA} />
            <KV label="Processed Size"   value="256 × 256 px (Attention U-Net input)" mono />
            <KV label="Total Pixels"     value={fmtNum(result.total_pixels)} mono />
            <KV label="MRI Modality"     value={study.mri_modality || study.modality} />
            <KV label="Ground Truth Mask" value={result.metrics && Object.values(result.metrics).some(v => v != null) ? 'Provided (evaluation)' : 'Not provided'} />
            <KV label="Inference Time"   value={fmtMs(result.inference_ms)} mono />
            <KV label="Pipeline Time"    value={fmtMs(result.total_ms)} mono />
          </div>
        </SectionCard>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          AI FINDINGS + CONFIDENCE  (balanced 2-col grid)
          ══════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-start">

        {/* ── Left ~33%: AI Segmentation Findings ── */}
        <div className="glass-card flex flex-col gap-4">
          <h3 className="section-title flex items-center gap-2 mb-0">
            <Activity className={`w-4 h-4 ${det ? 'text-red-500' : 'text-emerald-600'}`} />
            AI Segmentation Findings
          </h3>

          {/* Finding badge */}
          <div className={`flex items-start gap-3 p-3 rounded-xl border
                           ${det ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
            <div className={`p-2 rounded-lg flex-shrink-0
                             ${det ? 'bg-red-100' : 'bg-emerald-100'}`}>
              {det
                ? <Activity    className="w-4 h-4 text-red-500" />
                : <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
            </div>
            <div className="min-w-0">
              <p className={`text-sm font-semibold leading-snug
                             ${det ? 'text-red-700' : 'text-emerald-700'}`}>
                {det
                  ? 'Candidate Region Identified'
                  : 'No Candidate Tumour Region'}
              </p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                {det
                  ? 'AI-assisted finding — segmented region overlaid in figures below.'
                  : 'No candidate pixels exceeded the segmentation threshold.'}
              </p>
            </div>
          </div>

          {/* Disclaimer */}
          <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-100 pt-3">
            This is an AI-assisted research output and does not constitute a medical
            diagnosis. All results must be reviewed by a qualified radiologist.
          </p>
        </div>

        {/* ── Right ~67%: Tumour Probability & Confidence ── */}
        <div className="lg:col-span-2 glass-card flex flex-col gap-5">
          <h3 className="section-title flex items-center gap-2 mb-0">
            <Target className="w-4 h-4 text-purple-500" />
            Tumour Probability &amp; Confidence
          </h3>

          {/* 4-metric grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* Confidence Score */}
            <div className="metric-badge">
              <span className="label-text mb-2">Confidence Score</span>
              <span className={`inline-flex items-center px-2.5 py-1 rounded-full border
                               text-base font-bold font-mono
                               ${confidenceColour(result.confidence)}`}>
                {result.confidence != null
                  ? `${(result.confidence * 100).toFixed(1)}%`
                  : NA}
              </span>
            </div>

            {/* Tumour Area */}
            <div className="metric-badge">
              <span className="label-text mb-2">Tumour Area</span>
              <span className={`text-xl font-bold font-mono
                               ${det ? 'text-red-500' : 'text-emerald-600'}`}>
                {fmtPct(result.tumor_percentage)}
              </span>
            </div>

            {/* Tumour Pixels */}
            <div className="metric-badge">
              <span className="label-text mb-2">Tumour Pixels</span>
              <span className="text-xl font-bold font-mono text-slate-700">
                {fmtNum(result.tumor_pixels)}
              </span>
            </div>

            {/* Total Scan Pixels */}
            <div className="metric-badge">
              <span className="label-text mb-2">Total Scan Pixels</span>
              <span className="text-xl font-bold font-mono text-slate-700">
                {fmtNum(result.total_pixels)}
              </span>
            </div>
          </div>

          {/* Confidence progress bar */}
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1.5">
              <span>Confidence Level</span>
              <span className="font-mono font-semibold text-slate-600">
                {result.confidence != null
                  ? `${(result.confidence * 100).toFixed(1)}%`
                  : '—'}
              </span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  (result.confidence || 0) >= 0.7 ? 'bg-emerald-500'
                  : (result.confidence || 0) >= 0.5 ? 'bg-amber-400'
                  : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(100, (result.confidence || 0) * 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-300 mt-1">
              <span>Low</span>
              <span>Medium</span>
              <span>High</span>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          VISUALIZATION
          ══════════════════════════════════════════════════════════════════ */}
      <SectionCard title="Tumour Region Visualization" icon={Eye} iconTint="text-purple-400">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <ImageTile src={images.original}  label="Processed MRI" download="mri_processed.png" />
          <ImageTile src={images.pred_mask} label="Segmentation Mask" download="segmentation_mask.png" />
          <ImageTile src={images.overlay}   label="Segmentation Overlay" download="overlay.png" />
          <ImageTile src={images.prob_map}  label="Probability Map" download="probability_map.png" />
        </div>
        {images.uncertainty && (
          <div className="mt-4">
            <p className="label-text mb-3 flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5 text-purple-400" />
              Epistemic Uncertainty Map (MC-Dropout)
            </p>
            <div className="max-w-xs">
              <ImageTile
                src={images.uncertainty}
                label="Uncertainty"
                download="uncertainty.png"
              />
            </div>
            <p className="text-xs text-slate-400 mt-3">
              Brighter regions indicate higher model uncertainty. Low-confidence areas warrant
              additional specialist review.
            </p>
          </div>
        )}
      </SectionCard>

      {/* ══════════════════════════════════════════════════════════════════
          QUANTITATIVE METRICS
          ══════════════════════════════════════════════════════════════════ */}
      <MetricsPanel
        metrics={result.metrics}
        tumorPct={result.tumor_percentage}
        confidence={result.confidence}
        inferenceMs={result.inference_ms}
      />

      {/* ══════════════════════════════════════════════════════════════════
          LIMITATIONS
          ══════════════════════════════════════════════════════════════════ */}
      <SectionCard title="Limitations &amp; Review" icon={AlertTriangle} iconTint="text-amber-400">
        <ul className="space-y-2">
          {LIMITATIONS.map((line) => (
            <li key={line} className="flex items-start gap-2.5 text-sm text-slate-600">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-amber-400/70 shrink-0" />
              <span>{line}</span>
            </li>
          ))}
        </ul>
      </SectionCard>

      {/* ══════════════════════════════════════════════════════════════════
          DISCLAIMER FOOTER
          ══════════════════════════════════════════════════════════════════ */}
      <div className="text-center pt-2">
        <p className="text-xs text-slate-400">
          Brain Tumor Segmentation — Research tool only. AI-assisted output · Not a medical diagnosis · Requires radiologist review.
        </p>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          PDF PREVIEW MODAL
          ══════════════════════════════════════════════════════════════════ */}
      {previewOpen && activeBlob && (
        <PdfPreviewModal
          blob={activeBlob}
          filename={filenameFromResult()}
          onClose={() => setPreviewOpen(false)}
        />
      )}
    </div>
  )
}
