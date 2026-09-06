import React from 'react'
import { BarChart2, Target } from 'lucide-react'
import { fmtMetric, fmtMs, metricColour, confidenceColour } from '../utils/helpers'

function MetricTile({ label, value, formatter = fmtMetric, colourFn = metricColour, unit = '' }) {
  const formatted = value != null ? formatter(value) + unit : '—'
  const colour    = value != null ? colourFn(value) : 'text-slate-300'
  return (
    <div className="metric-badge">
      <span className="label-text mb-2">{label}</span>
      <span className={`text-xl font-bold font-mono ${colour}`}>{formatted}</span>
    </div>
  )
}

export default function MetricsPanel({ metrics, tumorPct, confidence, inferenceMs }) {
  const hasGtMetrics = metrics && Object.values(metrics).some((v) => v != null)

  return (
    <div className="glass-card space-y-6">
      <h3 className="section-title flex items-center gap-2">
        <BarChart2 className="w-4 h-4 text-medical-600" />
        Quantitative Analysis
      </h3>

      {/* Basic stats */}
      <div>
        <p className="label-text mb-3">Detection Summary</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <MetricTile
            label="Tumour Area"
            value={tumorPct}
            formatter={(v) => Number(v).toFixed(2)}
            unit="%"
            colourFn={(v) => v > 0 ? 'text-red-500' : 'text-emerald-600'}
          />
          <div className="metric-badge">
            <span className="label-text mb-2">Confidence</span>
            <span className={`text-xl font-bold font-mono ${
              confidence >= 0.7 ? 'text-emerald-600'
              : confidence >= 0.5 ? 'text-amber-500'
              : 'text-red-500'
            }`}>
              {confidence != null ? `${(confidence * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
          <MetricTile
            label="Inference Time"
            value={inferenceMs}
            formatter={fmtMs}
            colourFn={() => 'text-medical-600'}
          />
        </div>
      </div>

      {/* Ground-truth metrics */}
      {hasGtMetrics ? (
        <div>
          <p className="label-text mb-3 flex items-center gap-2">
            <Target className="w-3.5 h-3.5" />
            Performance vs. Ground Truth
          </p>
          <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            {[
              { key: 'dice',        label: 'Dice'        },
              { key: 'iou',         label: 'IoU'         },
              { key: 'precision',   label: 'Precision'   },
              { key: 'recall',      label: 'Recall'      },
              { key: 'f1',          label: 'F1'          },
              { key: 'sensitivity', label: 'Sensitivity' },
              { key: 'specificity', label: 'Specificity' },
            ].map(({ key, label }) => (
              <MetricTile key={key} label={label} value={metrics[key]} />
            ))}
          </div>
          <p className="text-xs text-slate-400 mt-3">
            Metrics computed against the provided ground-truth mask.
          </p>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
          <p className="text-sm text-slate-400 text-center">
            Upload a ground-truth mask alongside the MRI to compute
            Dice, IoU, Precision, Recall, Sensitivity and Specificity.
          </p>
        </div>
      )}
    </div>
  )
}
