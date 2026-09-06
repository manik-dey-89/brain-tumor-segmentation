import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Brain, Layers, BarChart2, Zap, Shield, Code2,
  ChevronRight, Activity, Target, Cpu, ArrowRight,
  Upload, ScanLine, FileImage, FlaskConical,
  CheckCircle2, GitBranch, Microscope, Sigma,
  Gauge, Database, Network, BookOpen,
} from 'lucide-react'

/* ─── Data ─────────────────────────────────────────────────────────────── */

const PLATFORM_STATS = [
  {
    icon: FileImage,
    label: 'Supported Formats',
    value: 'PNG · JPG · TIFF · NIfTI',
    sub: '2D slices and volumetric .nii/.nii.gz',
    colour: 'text-medical-400 bg-medical-500/10 border-medical-500/20',
  },
  {
    icon: Network,
    label: 'Segmentation Architecture',
    value: 'Attention U-Net',
    sub: 'Encoder–decoder with soft attention gates',
    colour: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  },
  {
    icon: Sigma,
    label: 'Evaluation Metrics',
    value: 'Dice · IoU · Hausdorff',
    sub: 'Plus Precision, Recall, F1, Sensitivity, Specificity',
    colour: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  },
  {
    icon: Gauge,
    label: 'Processing Pipeline',
    value: '5-Stage Inference',
    sub: 'Upload → Pre-process → Infer → Overlay → Export',
    colour: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
  },
]

const FEATURES = [
  {
    icon: Brain,
    title: 'Attention U-Net',
    desc: 'Encoder–decoder architecture with soft attention gates that learn to focus on tumour regions, suppressing irrelevant background activations.',
    colour: 'text-medical-400 bg-medical-500/10 border-medical-500/20',
  },
  {
    icon: Layers,
    title: 'Multi-Architecture Support',
    desc: 'Switch between U-Net, Attention U-Net and ResUNet from a single YAML config file — no code changes required.',
    colour: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  },
  {
    icon: BarChart2,
    title: 'Full Evaluation Suite',
    desc: 'Dice coefficient, IoU, Precision, Recall, F1, Sensitivity, Specificity and Hausdorff distance — computed automatically when a ground-truth mask is provided.',
    colour: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  },
  {
    icon: Shield,
    title: 'Uncertainty Estimation',
    desc: 'Monte Carlo Dropout produces per-pixel uncertainty maps so low-confidence prediction regions are explicitly flagged in results.',
    colour: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
  },
  {
    icon: Zap,
    title: 'TTA & Mixed Precision',
    desc: 'Test-time augmentation with horizontal/vertical flips and AMP (fp16) inference for improved robustness and GPU throughput.',
    colour: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
  },
  {
    icon: Code2,
    title: 'Production-Ready API',
    desc: 'FastAPI backend with full OpenAPI documentation, CORS, per-upload file validation, patient study metadata and persistent history tracking.',
    colour: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
  },
]

const PIPELINE = [
  {
    step: '01',
    label: 'Upload MRI',
    detail: 'PNG, JPG, TIFF or NIfTI · up to 50 MB',
    icon: Upload,
    colour: 'from-medical-600 to-medical-400',
    ring: 'ring-medical-500/30',
  },
  {
    step: '02',
    label: 'Pre-processing',
    detail: 'Normalise · resize · channel conversion',
    icon: Cpu,
    colour: 'from-purple-600 to-purple-400',
    ring: 'ring-purple-500/30',
  },
  {
    step: '03',
    label: 'U-Net Inference',
    detail: 'Attention U-Net forward pass · TTA optional',
    icon: Brain,
    colour: 'from-cyan-600 to-cyan-400',
    ring: 'ring-cyan-500/30',
  },
  {
    step: '04',
    label: 'Segmentation Overlay',
    detail: 'Binary mask · probability heatmap · composite',
    icon: Layers,
    colour: 'from-emerald-600 to-emerald-400',
    ring: 'ring-emerald-500/30',
  },
  {
    step: '05',
    label: 'Metrics & Export',
    detail: 'Quantitative report · PDF download',
    icon: Target,
    colour: 'from-yellow-600 to-yellow-400',
    ring: 'ring-yellow-500/30',
  },
]

const HOW_IT_WORKS = [
  {
    n: '1',
    title: 'Register Study',
    body: 'Enter patient ID, study metadata and modality before uploading, so every prediction is traceable and exportable.',
    icon: Database,
    colour: 'bg-medical-500/15 border-medical-500/30 text-medical-300',
  },
  {
    n: '2',
    title: 'Upload MRI Scan',
    body: 'Drag and drop a 2-D MRI slice or a full NIfTI volume. The platform automatically extracts the middle axial slice for 2-D segmentation.',
    icon: ScanLine,
    colour: 'bg-purple-500/15 border-purple-500/30 text-purple-300',
  },
  {
    n: '3',
    title: 'Automated Segmentation',
    body: 'The Attention U-Net model runs inference, generates a binary tumour mask, a probability heatmap, and an alpha-blended overlay in a single pass.',
    icon: Brain,
    colour: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300',
  },
  {
    n: '4',
    title: 'Review & Export',
    body: 'Inspect segmentation results, quantitative metrics and uncertainty maps. Download a structured A4 PDF research report.',
    icon: BookOpen,
    colour: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
  },
]

const RESEARCH_CARDS = [
  {
    icon: Microscope,
    title: 'Semantic Segmentation',
    body: 'Pixel-wise classification using fully-convolutional encoder–decoder networks. Attention gates learn spatial weighting to improve tumour boundary precision without post-processing heuristics.',
    colour: 'text-medical-400',
  },
  {
    icon: BarChart2,
    title: 'Quantitative Analysis',
    body: 'Segmentation quality is measured using Dice coefficient (overlap agreement) and Hausdorff distance (boundary error), alongside Precision, Recall and IoU when reference masks are available.',
    colour: 'text-emerald-400',
  },
  {
    icon: FlaskConical,
    title: 'Uncertainty Estimation',
    body: 'Monte Carlo Dropout approximates Bayesian inference at test time. Multiple stochastic forward passes produce a variance map that highlights voxels where the model lacks confidence.',
    colour: 'text-yellow-400',
  },
]

/* ─── SVG MRI brain visual ───────────────────────────────────────────── */

function MriBrainVisual() {
  return (
    <div className="relative w-full h-full flex items-center justify-center select-none pointer-events-none">
      {/* Outer glow ring */}
      <div className="absolute w-72 h-72 sm:w-80 sm:h-80 rounded-full
                      border border-medical-500/10
                      bg-radial-gradient"
           style={{ background: 'radial-gradient(circle, rgba(45,142,255,0.08) 0%, transparent 70%)' }} />

      {/* Pulse rings */}
      <div className="absolute w-56 h-56 rounded-full border border-medical-400/10 animate-[pulse_4s_ease-in-out_infinite]" />
      <div className="absolute w-72 h-72 rounded-full border border-medical-400/5  animate-[pulse_5s_ease-in-out_infinite_0.8s]" />

      {/* Central brain SVG — simple schematic illustration */}
      <div className="float-anim relative z-10">
        <svg
          width="220" height="220" viewBox="0 0 220 220"
          fill="none" xmlns="http://www.w3.org/2000/svg"
          opacity="0.82"
        >
          {/* Outer skull silhouette */}
          <ellipse cx="110" cy="108" rx="82" ry="88"
                   fill="none" stroke="rgba(45,142,255,0.18)" strokeWidth="1.2" />

          {/* Brain hemisphere L */}
          <path d="M110 50 C72 52 48 72 46 100 C44 124 56 148 74 158 C88 166 102 162 110 155"
                fill="none" stroke="rgba(45,142,255,0.35)" strokeWidth="1.5" strokeLinecap="round" />
          {/* Brain hemisphere R */}
          <path d="M110 50 C148 52 172 72 174 100 C176 124 164 148 146 158 C132 166 118 162 110 155"
                fill="none" stroke="rgba(45,142,255,0.35)" strokeWidth="1.5" strokeLinecap="round" />
          {/* Corpus callosum */}
          <path d="M82 96 C92 90 118 90 138 96"
                fill="none" stroke="rgba(45,142,255,0.28)" strokeWidth="1.2" strokeLinecap="round" />
          {/* Gyri lines L */}
          <path d="M58 86 C64 78 76 76 82 82"  fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M52 104 C58 96 72 93 80 99" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M54 122 C62 114 76 113 82 119" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M62 138 C72 132 86 131 90 137" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          {/* Gyri lines R */}
          <path d="M162 86 C156 78 144 76 138 82"  fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M168 104 C162 96 148 93 140 99" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M166 122 C158 114 144 113 138 119" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />
          <path d="M158 138 C148 132 134 131 130 137" fill="none" stroke="rgba(80,173,255,0.22)" strokeWidth="1" />

          {/* Tumour region highlight */}
          <ellipse cx="136" cy="82" rx="18" ry="14"
                   fill="rgba(239,68,68,0.12)" stroke="rgba(239,68,68,0.45)"
                   strokeWidth="1.2" strokeDasharray="3 2" />
          <ellipse cx="136" cy="82" rx="9" ry="7"
                   fill="rgba(239,68,68,0.20)" />

          {/* Segmentation overlay dots */}
          <circle cx="128" cy="78" r="2" fill="rgba(239,68,68,0.7)" />
          <circle cx="138" cy="76" r="2.5" fill="rgba(239,68,68,0.8)" />
          <circle cx="144" cy="82" r="1.8" fill="rgba(239,68,68,0.65)" />
          <circle cx="136" cy="88" r="2" fill="rgba(239,68,68,0.6)" />

          {/* Crosshair on tumour */}
          <line x1="136" y1="70" x2="136" y2="94" stroke="rgba(239,68,68,0.4)" strokeWidth="0.8" />
          <line x1="120" y1="82" x2="152" y2="82" stroke="rgba(239,68,68,0.4)" strokeWidth="0.8" />

          {/* Scan lines (horizontal) */}
          {[60, 76, 92, 108, 124, 140, 155].map((y, i) => (
            <line key={i} x1="40" x2="180" y1={y} y2={y}
                  stroke="rgba(45,142,255,0.06)" strokeWidth="0.8" />
          ))}

          {/* Probability heatmap bar at bottom */}
          <rect x="60" y="172" width="100" height="8" rx="4"
                fill="url(#heatGrad)" opacity="0.7" />
          <text x="57" y="192" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace">0.0</text>
          <text x="148" y="192" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace">1.0</text>
          <text x="95" y="192" fill="rgba(255,255,255,0.25)" fontSize="7" fontFamily="monospace">P(tumour)</text>

          <defs>
            <linearGradient id="heatGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#1e3a5f" />
              <stop offset="40%"  stopColor="#2d8eff" />
              <stop offset="70%"  stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
        </svg>

        {/* Floating annotation badges */}
        <div className="absolute -top-2 -right-4 flex items-center gap-1.5
                        bg-red-500/15 border border-red-500/30 rounded-lg px-2.5 py-1.5">
          <span className="w-2 h-2 rounded-full bg-red-400" />
          <span className="text-xs font-medium text-red-300">Candidate Region</span>
        </div>
        <div className="absolute bottom-6 -left-6 flex items-center gap-1.5
                        bg-medical-500/15 border border-medical-500/30 rounded-lg px-2.5 py-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-medical-400" />
          <span className="text-xs font-medium text-medical-300">Mask Generated</span>
        </div>
      </div>
    </div>
  )
}

/* ─── Background slideshow ──────────────────────────────────────────── */

const BG_IMAGES = ['/bg-no1.png', '/bg-no2.png']
const SLIDE_DURATION = 5000   // ms each image stays visible
const FADE_DURATION  = 1400   // ms crossfade

function HeroBgSlideshow() {
  const [current, setCurrent] = useState(0)
  const [next,    setNext]    = useState(1)
  const [nextVis, setNextVis] = useState(false)

  useEffect(() => {
    const timer = setInterval(() => {
      setNextVis(true)
      setTimeout(() => {
        setCurrent(c => (c + 1) % BG_IMAGES.length)
        setNext(n    => (n + 1) % BG_IMAGES.length)
        setNextVis(false)
      }, FADE_DURATION)
    }, SLIDE_DURATION)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
      {/* Current image — full visibility */}
      <img
        key={`cur-${current}`}
        src={BG_IMAGES[current]}
        alt=""
        className="absolute inset-0 w-full h-full object-cover object-center"
        style={{
          opacity: 1,
          filter: 'brightness(0.75)',
          transition: `opacity ${FADE_DURATION}ms ease-in-out`,
        }}
      />
      {/* Next image — crossfades in */}
      <img
        key={`nxt-${next}`}
        src={BG_IMAGES[next]}
        alt=""
        className="absolute inset-0 w-full h-full object-cover object-center"
        style={{
          opacity: nextVis ? 1 : 0,
          filter: 'brightness(0.75)',
          transition: `opacity ${FADE_DURATION}ms ease-in-out`,
        }}
      />
      {/* Left-side dark gradient so text is readable */}
      <div className="absolute inset-0"
           style={{
             background: 'linear-gradient(to right, rgba(6,13,31,0.92) 0%, rgba(6,13,31,0.70) 38%, rgba(6,13,31,0.15) 65%, transparent 100%)',
           }} />
      {/* Top dark strip so navbar contrast is preserved */}
      <div className="absolute top-0 inset-x-0 h-24"
           style={{ background: 'linear-gradient(to bottom, rgba(6,13,31,0.6), transparent)' }} />
      {/* Bottom fade into page */}
      <div className="absolute bottom-0 inset-x-0 h-48"
           style={{ background: 'linear-gradient(to bottom, transparent, #060d1f)' }} />
    </div>
  )
}

/* ─── Pipeline connector ─────────────────────────────────────────────── */

function PipelineConnector({ delay = 0 }) {
  return (
    <div className="hidden sm:flex items-center flex-shrink-0 mx-1">
      <div className="relative w-12 h-px">
        <div className="absolute inset-0 bg-gradient-to-r from-white/5 via-medical-400/30 to-white/5
                        flow-connector rounded-full"
             style={{ animationDelay: `${delay}s` }} />
        <div className="absolute right-0 top-1/2 -translate-y-1/2
                        w-1.5 h-1.5 rounded-full bg-medical-400/40" />
      </div>
    </div>
  )
}

/* ─── Page ────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  return (
    <div className="overflow-x-hidden">

      {/* ══════════════════════════════════════════════════════════════════
          HERO
          ══════════════════════════════════════════════════════════════════ */}
      <section className="relative min-h-[calc(100vh-4rem)] flex items-center">

        {/* Full-bleed background slideshow */}
        <HeroBgSlideshow />

        {/* Content — left-aligned, max half-width on desktop */}
        <div className="relative w-full max-w-7xl mx-auto px-4 sm:px-8 py-20 lg:py-28">
          <div className="max-w-xl space-y-7">

            {/* Badge */}
            <div className="inline-flex items-center gap-2 w-fit px-3.5 py-1.5 rounded-full
                            bg-medical-600/15 border border-medical-500/30
                            text-medical-300 text-xs font-medium backdrop-blur-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-medical-400 animate-pulse" />
              Research-Grade Brain Tumour Segmentation
            </div>

            {/* Headline */}
            <h1 className="text-5xl sm:text-6xl xl:text-7xl font-bold tracking-tight leading-[1.05]">
              <span className="text-white drop-shadow-lg">Deep Learning</span>
              <br />
              <span className="bg-gradient-to-r from-medical-400 via-blue-400 to-purple-400
                               bg-clip-text text-transparent drop-shadow-lg">
                MRI Tumour
              </span>
              <br />
              <span className="text-white drop-shadow-lg">Segmentation</span>
            </h1>

            {/* Sub */}
            <p className="text-base sm:text-lg text-white/65 leading-relaxed">
              End-to-end AI pipeline — upload an MRI scan and receive a segmentation mask,
              probability heatmap, overlay visualisation and quantitative evaluation metrics.
              Built on Attention U-Net with FastAPI and React.
            </p>

            {/* Disclaimer chip */}
            <p className="text-xs text-yellow-300/80 bg-yellow-500/10 border border-yellow-500/20
                          rounded-lg px-3 py-2 w-fit backdrop-blur-sm">
              ⚠ Research tool only — not a medical device or diagnostic instrument.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap items-center gap-3">
              <Link to="/analyze"
                    className="btn-primary flex items-center gap-2 text-sm sm:text-base
                               px-6 py-3 sm:px-8 sm:py-3.5 shadow-xl shadow-medical-900/60">
                <Brain className="w-4 h-4 sm:w-5 sm:h-5" />
                Start Analysis
                <ChevronRight className="w-4 h-4" />
              </Link>
              <Link to="/history"
                    className="btn-secondary flex items-center gap-2 text-sm sm:text-base
                               backdrop-blur-sm">
                <Activity className="w-4 h-4" />
                View History
              </Link>
            </div>

            {/* Quick-stat strip */}
            <div className="flex flex-wrap gap-6 pt-2 border-t border-white/10">
              {[
                { label: 'Architecture', val: 'Attention U-Net' },
                { label: 'Backend',      val: 'FastAPI + PyTorch' },
                { label: 'Output',       val: 'Mask · Heatmap · PDF' },
              ].map(({ label, val }) => (
                <div key={label} className="text-left">
                  <p className="text-[10px] uppercase tracking-wider text-white/35">{label}</p>
                  <p className="text-sm font-semibold text-white/80">{val}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          PLATFORM OVERVIEW
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {PLATFORM_STATS.map(({ icon: Icon, label, value, sub, colour }) => (
            <div key={label}
                 className="glass-card hover:shadow-md transition-all group p-5">
              <div className={`w-10 h-10 rounded-xl border flex items-center
                               justify-center mb-3 ${colour} flex-shrink-0`}>
                <Icon className="w-5 h-5" />
              </div>
              <p className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">{label}</p>
              <p className="text-sm font-bold text-slate-800 leading-snug">{value}</p>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          END-TO-END PIPELINE
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="text-center mb-10">
          <p className="text-xs uppercase tracking-widest text-medical-500/80 mb-2">
            Inference Workflow
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">End-to-End Pipeline</h2>
          <p className="text-slate-400 mt-2 text-sm max-w-xl mx-auto">
            Five automated stages from raw MRI to structured research report
          </p>
        </div>

        {/* Pipeline row */}
        <div className="relative flex flex-col sm:flex-row items-stretch justify-center gap-3 sm:gap-0">
          {PIPELINE.map(({ step, label, detail, icon: Icon, colour, ring }, idx) => (
            <React.Fragment key={step}>
              <div className={`flex flex-col items-center gap-2.5 text-center
                               sm:w-36 lg:w-40 group`}>
                {/* Icon bubble */}
                <div className={`relative w-14 h-14 rounded-2xl bg-gradient-to-br ${colour}
                                 flex items-center justify-center shadow-lg
                                 ring-4 ${ring}
                                 group-hover:scale-105 transition-transform duration-200`}>
                  <Icon className="w-6 h-6 text-white" />
                  {/* Step badge */}
                  <span className="absolute -top-2 -right-2 w-5 h-5 rounded-full
                                   bg-white border border-slate-200
                                   text-[9px] font-bold text-slate-400
                                   flex items-center justify-center shadow-sm">
                    {idx + 1}
                  </span>
                </div>
                <p className="text-xs sm:text-sm font-semibold text-slate-700">{label}</p>
                <p className="text-[11px] text-slate-400 leading-tight px-1">{detail}</p>
              </div>

              {idx < PIPELINE.length - 1 && (
                <PipelineConnector delay={idx * 0.4} />
              )}
            </React.Fragment>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          HOW ANALYSIS WORKS
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="text-center mb-10">
          <p className="text-xs uppercase tracking-widest text-purple-500/80 mb-2">
            Step-by-Step Workflow
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">How Analysis Works</h2>
          <p className="text-slate-400 mt-2 text-sm">
            From patient registration to segmentation report in four steps
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {HOW_IT_WORKS.map(({ n, title, body, icon: Icon, colour }, idx) => (
            <div key={n}
                 className="glass-card hover:shadow-md transition-all group relative overflow-hidden p-6">
              <span className="absolute -top-3 -right-1 text-7xl font-black text-slate-100
                               select-none pointer-events-none">
                {n}
              </span>
              <div className={`w-10 h-10 rounded-xl border flex items-center
                               justify-center mb-4 ${colour} flex-shrink-0`}>
                <Icon className="w-5 h-5" />
              </div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono text-slate-300">0{n}</span>
                <h3 className="font-semibold text-slate-800 text-sm">{title}</h3>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          CORE FEATURES
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="text-center mb-10">
          <p className="text-xs uppercase tracking-widest text-cyan-600/80 mb-2">
            Technical Capabilities
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Core Features</h2>
          <p className="text-slate-400 mt-2 text-sm max-w-xl mx-auto">
            Attention U-Net segmentation with structured evaluation and production-ready tooling
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, title, desc, colour }) => (
            <div key={title}
                 className="glass-card hover:shadow-md transition-all duration-200
                            group hover:-translate-y-0.5 p-6">
              <div className={`w-11 h-11 rounded-xl border flex items-center
                               justify-center mb-4 ${colour}
                               group-hover:scale-105 transition-transform duration-200`}>
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-800 mb-2 text-sm">{title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          RESEARCH & VALIDATION
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="text-center mb-10">
          <p className="text-xs uppercase tracking-widest text-emerald-600/80 mb-2">
            Methodology
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Research & Validation</h2>
          <p className="text-slate-400 mt-2 text-sm max-w-xl mx-auto">
            Grounded in established deep learning techniques for medical image segmentation
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-6">
          {RESEARCH_CARDS.map(({ icon: Icon, title, body, colour }) => (
            <div key={title} className="glass-card hover:shadow-md transition-colors p-6">
              <Icon className={`w-7 h-7 ${colour} mb-4`} />
              <h3 className="font-semibold text-slate-800 text-sm mb-2">{title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5
                        grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            {
              label: 'Input Modality',
              val: '2-D MRI slices (axial) or NIfTI volumes — middle slice extracted automatically',
            },
            {
              label: 'Loss Function',
              val: 'Composite Binary Cross-Entropy + Dice loss for class-imbalanced tumour segmentation',
            },
            {
              label: 'Output Artefacts',
              val: 'Binary mask, probability heatmap (float32), alpha-composite overlay, PDF report',
            },
          ].map(({ label, val }) => (
            <div key={label}>
              <p className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">{label}</p>
              <p className="text-xs text-slate-600 leading-relaxed">{val}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          CTA
          ══════════════════════════════════════════════════════════════════ */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-20">
        <div className="relative overflow-hidden bg-gradient-to-br from-medical-600 to-medical-800
                        rounded-2xl text-center py-16 px-8 shadow-lg shadow-medical-200">
          {/* Background decoration */}
          <div className="absolute inset-0 pointer-events-none opacity-10" aria-hidden>
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-32"
                 style={{ background: 'radial-gradient(ellipse, white 0%, transparent 70%)' }} />
          </div>

          <div className="relative z-10 max-w-2xl mx-auto space-y-5">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-white/20 border border-white/30
                            flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>

            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Ready to run segmentation?
            </h2>
            <p className="text-white/75 text-sm sm:text-base leading-relaxed">
              Upload an MRI scan, register study metadata, and receive a structured
              segmentation report with quantitative metrics — all in your browser.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
              <Link to="/analyze"
                    className="flex items-center gap-2 text-base px-8 py-3.5 rounded-xl
                               bg-white text-medical-700 font-semibold
                               hover:bg-slate-50 transition-colors shadow-md
                               active:scale-95">
                <Brain className="w-5 h-5" />
                Start Analysis
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/history"
                    className="flex items-center gap-2 px-6 py-3.5 rounded-xl
                               bg-white/15 border border-white/30 text-white font-medium
                               hover:bg-white/25 transition-colors active:scale-95">
                <GitBranch className="w-4 h-4" />
                Browse History
              </Link>
            </div>

            <p className="text-xs text-white/40 pt-2">
              Research and educational use only · Not a medical device · Results require radiologist review
            </p>
          </div>
        </div>
      </section>

    </div>
  )
}
