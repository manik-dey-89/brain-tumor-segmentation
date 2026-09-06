import React, { useState } from 'react'
import {
  Download, ZoomIn, Layers, GitCompare,
  Activity, AlertTriangle, CheckCircle2, Eye
} from 'lucide-react'
import { downloadBase64 } from '../utils/helpers'
import MetricsPanel from './MetricsPanel'

const VIEWS = [
  { id: 'original',  label: 'MRI',        icon: Eye     },
  { id: 'pred_mask', label: 'Mask',        icon: Layers  },
  { id: 'overlay',   label: 'Overlay',     icon: Activity},
  { id: 'prob_map',  label: 'Probability', icon: Activity},
]

function ImageCard({ src, label, onDownload, downloadName }) {
  const [zoom, setZoom] = useState(false)

  return (
    <div className="group relative glass rounded-2xl overflow-hidden">
      <div className="relative aspect-square bg-black/40">
        {src ? (
          <img
            src={src}
            alt={label}
            className="w-full h-full object-contain cursor-zoom-in"
            onClick={() => setZoom(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-white/20 text-sm">
            Not available
          </div>
        )}

        {/* Hover controls */}
        {src && (
          <div className="absolute top-2 right-2 flex gap-1.5 opacity-0
                          group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => setZoom(true)}
              className="p-1.5 rounded-lg bg-black/60 backdrop-blur-sm
                         hover:bg-black/80 transition-colors"
            >
              <ZoomIn className="w-3.5 h-3.5 text-white" />
            </button>
            {onDownload && (
              <button
                onClick={onDownload}
                className="p-1.5 rounded-lg bg-black/60 backdrop-blur-sm
                           hover:bg-black/80 transition-colors"
              >
                <Download className="w-3.5 h-3.5 text-white" />
              </button>
            )}
          </div>
        )}
      </div>
      <div className="px-3 py-2 border-t border-white/10">
        <p className="text-xs font-medium text-white/60 uppercase tracking-wider">{label}</p>
      </div>

      {/* Zoom modal */}
      {zoom && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setZoom(false)}
        >
          <img
            src={src}
            alt={label}
            className="max-w-full max-h-full object-contain rounded-xl"
          />
          <p className="absolute bottom-6 text-white/50 text-sm">Click anywhere to close</p>
        </div>
      )}
    </div>
  )
}

export default function ResultViewer({ result }) {
  const [activeView, setActiveView] = useState('overlay')

  if (!result) return null

  const { images, tumor_detected, tumor_percentage, confidence,
          metrics, disclaimer, filename, inference_ms, contours } = result

  const downloadImage = (key, name) => {
    if (images?.[key]) downloadBase64(images[key], name)
  }

  return (
    <div className="space-y-6 animate-slide-up">
      {/* ── Status banner ─────────────────────────────────────────── */}
      <div className={`flex items-start gap-4 p-5 rounded-2xl border ${
        tumor_detected
          ? 'bg-red-500/10 border-red-500/30'
          : 'bg-emerald-500/10 border-emerald-500/30'
      }`}>
        <div className={`p-2 rounded-xl flex-shrink-0 ${
          tumor_detected ? 'bg-red-500/20' : 'bg-emerald-500/20'
        }`}>
          {tumor_detected
            ? <AlertTriangle className="w-6 h-6 text-red-400" />
            : <CheckCircle2  className="w-6 h-6 text-emerald-400" />}
        </div>
        <div className="flex-1 min-w-0">
          <p className={`font-semibold text-lg ${
            tumor_detected ? 'text-red-300' : 'text-emerald-300'
          }`}>
            {tumor_detected
              ? `Tumour Region Detected — ${Number(tumor_percentage).toFixed(2)}% of scan area`
              : 'No Tumour Region Detected'}
          </p>
          <p className="text-sm text-white/50 mt-1">
            Confidence: {(confidence * 100).toFixed(1)}% · {filename} · {Math.round(inference_ms)}ms
          </p>
        </div>
      </div>

      {/* ── Image grid ────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {VIEWS.map(({ id, label }) => (
          <ImageCard
            key={id}
            src={images?.[id]}
            label={label}
            onDownload={images?.[id] ? () => downloadImage(id, `${id}.png`) : null}
            downloadName={`${id}.png`}
          />
        ))}
      </div>

      {/* ── Uncertainty map (if available) ────────────────────────── */}
      {images?.uncertainty && (
        <div className="glass-card">
          <h3 className="section-title flex items-center gap-2">
            <Activity className="w-4 h-4 text-purple-400" />
            Uncertainty Map (MC-Dropout)
          </h3>
          <div className="max-w-xs">
            <ImageCard
              src={images.uncertainty}
              label="Epistemic Uncertainty"
              onDownload={() => downloadImage('uncertainty', 'uncertainty.png')}
            />
          </div>
          <p className="text-xs text-white/40 mt-3">
            Brighter regions indicate higher model uncertainty.
            Low-confidence areas should be reviewed by a specialist.
          </p>
        </div>
      )}

      {/* ── Metrics ───────────────────────────────────────────────── */}
      <MetricsPanel
        metrics={metrics}
        tumorPct={tumor_percentage}
        confidence={confidence}
        inferenceMs={inference_ms}
      />

      {/* ── Download actions ──────────────────────────────────────── */}
      <div className="glass-card">
        <h3 className="section-title flex items-center gap-2">
          <Download className="w-4 h-4 text-medical-400" />
          Download Results
        </h3>
        <div className="flex flex-wrap gap-3">
          {[
            { key: 'pred_mask', label: 'Segmentation Mask', name: 'pred_mask.png' },
            { key: 'overlay',   label: 'Overlay Image',     name: 'overlay.png'   },
            { key: 'prob_map',  label: 'Probability Map',   name: 'prob_map.png'  },
          ].map(({ key, label, name }) =>
            images?.[key] ? (
              <button
                key={key}
                onClick={() => downloadImage(key, name)}
                className="btn-secondary flex items-center gap-2 text-sm"
              >
                <Download className="w-4 h-4" />
                {label}
              </button>
            ) : null
          )}
          <button
            onClick={() => {
              const report = {
                filename,
                tumor_detected,
                tumor_percentage,
                confidence,
                inference_ms,
                metrics,
                disclaimer,
              }
              const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url; a.download = 'report.json'
              a.click(); URL.revokeObjectURL(url)
            }}
            className="btn-secondary flex items-center gap-2 text-sm"
          >
            <Download className="w-4 h-4" />
            JSON Report
          </button>
        </div>
      </div>

      {/* ── Disclaimer ───────────────────────────────────────────── */}
      <div className="flex items-start gap-3 p-4 rounded-xl
                      bg-yellow-500/10 border border-yellow-500/20">
        <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-yellow-300/80">{disclaimer}</p>
      </div>
    </div>
  )
}
