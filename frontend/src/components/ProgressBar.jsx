import React from 'react'
import { Loader2 } from 'lucide-react'

const STAGES = [
  { min: 0,  max: 50,  label: 'Uploading image…'        },
  { min: 50, max: 70,  label: 'Preprocessing MRI…'       },
  { min: 70, max: 90,  label: 'Running U-Net inference…' },
  { min: 90, max: 100, label: 'Generating results…'      },
]

function getStageLabel(progress) {
  const stage = STAGES.find((s) => progress >= s.min && progress < s.max)
  return stage?.label || 'Processing…'
}

export default function ProgressBar({ progress, status }) {
  if (status === 'idle' || status === 'done' || status === 'error') return null

  return (
    <div className="glass-card animate-fade-in">
      <div className="flex items-center gap-3 mb-4">
        <Loader2 className="w-5 h-5 text-medical-400 animate-spin" />
        <span className="text-sm font-medium text-white/80">
          {getStageLabel(progress)}
        </span>
        <span className="ml-auto text-sm font-mono text-medical-400 font-bold">
          {progress}%
        </span>
      </div>

      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-medical-600 to-medical-400
                     transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="flex gap-4 mt-4">
        {STAGES.map((stage, i) => {
          const done    = progress > stage.max
          const active  = progress >= stage.min && progress < stage.max
          return (
            <div key={i} className="flex items-center gap-1.5 text-xs">
              <span className={`w-1.5 h-1.5 rounded-full transition-colors ${
                done    ? 'bg-emerald-400'
                : active ? 'bg-medical-400 animate-pulse'
                : 'bg-white/20'
              }`} />
              <span className={done ? 'text-emerald-400' : active ? 'text-white/70' : 'text-white/30'}>
                {stage.label.replace('…', '')}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
