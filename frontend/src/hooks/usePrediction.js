/**
 * usePrediction – custom hook managing the full upload → predict → report workflow
 * with explicit 6-stage pipeline feedback.
 *
 * Stage states: 'pending' | 'active' | 'done' | 'error'
 *
 * Stages 0–5 (Upload, Preprocessing, Inference, Segmentation, Quantitative)
 * are inferred from the single /predict HTTP call (backend performs them).
 * Stage 6 (Report Generation) is executed client-side via a separate HTTP call.
 */
import { useState, useCallback, useRef } from 'react'
import toast from 'react-hot-toast'
import { predict as apiPredict, generateReportPdf, wakeBackend } from '../utils/api'
import { STAGES } from '../components/AnalysisPipeline'

const INITIAL_STAGES = Object.fromEntries(STAGES.map(s => [s.id, 'pending']))

const INITIAL_STATE = {
  status:          'idle',
  stages:          { ...INITIAL_STAGES },
  uploadProgress:  0,
  overallProgress: 0,
  result:          null,
  error:           null,
  imageFile:       null,
  maskFile:        null,
  studyMeta:       null,
  startTime:       null,
  pdfBlob:         null,
  pdfError:        null,
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */

function recomputeOverallProgress(stages) {
  let done = 0
  let active = 0
  for (const s of STAGES) {
    const st = stages[s.id] || 'pending'
    if (st === 'done') done++
    else if (st === 'active') active = 0.5
  }
  const base = ((done + active) / STAGES.length) * 100
  return Math.min(100, Math.round(base))
}

export default function usePrediction() {
  const [state, setState] = useState(INITIAL_STATE)
  const timerRef = useRef(null)
  const cancelledRef = useRef(false)
  const pdfAttemptRef = useRef(0)

  /* ── Low-level state setters ────────────────────────────────────────── */

  const setField = useCallback((patch) => {
    setState(prev => ({ ...prev, ...patch }))
  }, [])

  const setStage = useCallback((stageId, status) => {
    setState(prev => {
      const stages = { ...prev.stages, [stageId]: status }
      return {
        ...prev,
        stages,
        overallProgress: recomputeOverallProgress(stages),
      }
    })
  }, [])

  const clearTimers = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  /* ── File / meta handlers ──────────────────────────────────────────── */

  const setImageFile = useCallback((file) => {
    setState(prev => {
      if (prev.previewUrl) {
        try { URL.revokeObjectURL(prev.previewUrl) } catch {}
      }
      return {
        ...INITIAL_STATE,
        studyMeta: prev.studyMeta,
        imageFile: file,
        status: file ? 'upload_ready' : 'idle',
      }
    })
  }, [])

  const setMaskFile = useCallback((file) => {
    setField({ maskFile: file })
  }, [setField])

  const setStudyMeta = useCallback((meta) => {
    setField({ studyMeta: meta })
  }, [setField])

  const reset = useCallback(() => {
    cancelledRef.current = true
    clearTimers()
    if (timerRef.current) clearTimeout(timerRef.current)
    setState(INITIAL_STATE)
  }, [clearTimers])

  /* ── Staged UI – sequentially mark backend stages as done ──────────── */

  const playBackendStages = useCallback((onAllDone) => {
    // These are the stages that happened on the server inside /predict
    const backend = STAGES.slice(0, 5) // Upload, Preprocess, Inference, Segment, Quant
    let i = 0
    const runNext = () => {
      if (cancelledRef.current) return
      if (i >= backend.length) {
        onAllDone && onAllDone()
        return
      }
      const s = backend[i]
      setState(prev => {
        // Only mark ACTIVE if not already done (upload is done from progress)
        const cur = prev.stages[s.id] || 'pending'
        if (cur === 'done') {
          return prev
        }
        const stages = { ...prev.stages, [s.id]: 'active' }
        return { ...prev, stages, overallProgress: recomputeOverallProgress(stages) }
      })
      const dur = 220 + Math.round(Math.random() * 180) + (STAGES.indexOf(s) > 1 ? 160 : 0)
      timerRef.current = setTimeout(() => {
        setState(prev => {
          const stages = { ...prev.stages, [s.id]: 'done' }
          return { ...prev, stages, overallProgress: recomputeOverallProgress(stages) }
        })
        i++
        runNext()
      }, dur)
    }
    runNext()
  }, [])

  /* ── Generate PDF report (Stage 6 – actual work) ───────────────────── */

  const runReportGeneration = useCallback(async (resultData) => {
    if (cancelledRef.current) return { ok: false, cancelled: true }
    setStage('report', 'active')

    // Keep progress between 83% and 99% while report runs
    let progressPulseTimer = null
    let pulseProgress = 84
    progressPulseTimer = setInterval(() => {
      if (cancelledRef.current) {
        clearInterval(progressPulseTimer)
        return
      }
      pulseProgress = Math.min(99, pulseProgress + 1)
      setState(prev => ({
        ...prev,
        overallProgress: Math.max(prev.overallProgress, pulseProgress),
      }))
    }, 500)

    try {
      console.debug('[usePrediction] Generating PDF report… attempt', ++pdfAttemptRef.current)
      const blob = await generateReportPdf({ prediction_result: resultData })
      clearInterval(progressPulseTimer)

      if (cancelledRef.current) return { ok: false, cancelled: true }

      setStage('report', 'done')
      setState(prev => ({
        ...prev,
        pdfBlob: blob,
        pdfError: null,
        overallProgress: 100,
      }))
      console.debug('[usePrediction] PDF report generated successfully')
      return { ok: true, blob }
    } catch (err) {
      clearInterval(progressPulseTimer)
      if (cancelledRef.current) return { ok: false, cancelled: true }

      const msg = err.message || 'PDF generation failed'
      console.error('[usePrediction] PDF report generation failed:', msg)

      setStage('report', 'error')
      setState(prev => ({
        ...prev,
        pdfBlob: null,
        pdfError: msg,
        // keep overallProgress at whatever it was; don't reset success bar
      }))
      return { ok: false, error: msg }
    }
  }, [setStage])

  const retryReport = useCallback(async () => {
    if (!state.result) {
      toast.error('No analysis result available for report generation.')
      return
    }
    setState(prev => ({
      ...prev,
      pdfError: null,
    }))
    const res = await runReportGeneration(state.result)
    if (res.ok) {
      toast.success('PDF report generated')
    } else if (res.error) {
      toast.error(res.error)
    }
  }, [state.result, runReportGeneration])

  /* ── Main workflow entry point ─────────────────────────────────────── */

  const runPrediction = useCallback(async () => {
    if (!state.imageFile) {
      toast.error('Please upload an MRI scan first.')
      return
    }
    cancelledRef.current = false
    clearTimers()

    const startTime = Date.now()
    pdfAttemptRef.current = 0

    setState({
      ...INITIAL_STATE,
      status: 'uploading',
      imageFile: state.imageFile,
      maskFile: state.maskFile,
      studyMeta: state.studyMeta,
      startTime,
    })
    setStage('upload', 'active')

    let stagesStarted = false
    const startBackendStagePlayback = () => {
      if (stagesStarted) return
      stagesStarted = true
      playBackendStages()
    }
    const safetyTimer = setTimeout(() => {
      setState(prev => {
        if (prev.status === 'uploading' && prev.uploadProgress > 0) {
          if ((prev.stages.upload || 'pending') !== 'done') {
            const stages = { ...prev.stages, upload: 'done' }
            setField({ stages, overallProgress: recomputeOverallProgress(stages) })
          }
          startBackendStagePlayback()
        }
        return prev
      })
    }, 2000)

    const uploadProgressFn = (pct) => {
      if (cancelledRef.current) return
      const clamped = Math.min(50, Math.round(pct))
      setState(prev => {
        const stages = { ...prev.stages }
        if (clamped >= 50 && stages.upload !== 'done') stages.upload = 'done'
        return {
          ...prev,
          uploadProgress: clamped,
          stages,
          overallProgress: Math.max(prev.overallProgress, recomputeOverallProgress(stages)),
        }
      })
    }

    setField({ status: 'processing' })

    // Wake the backend from Render free-tier sleep before sending the heavy
    // predict payload — absorbs cold-start latency outside the main request.
    await wakeBackend()

    try {
      const res = await apiPredict(
        state.imageFile,
        state.maskFile || null,
        state.studyMeta || null,
        uploadProgressFn,
      )

      if (cancelledRef.current) return
      clearTimeout(safetyTimer)

      // Ensure upload is marked done
      setState(prev => {
        if ((prev.stages.upload || 'pending') !== 'done') {
          const stages = { ...prev.stages, upload: 'done' }
          return { ...prev, stages, overallProgress: recomputeOverallProgress(stages) }
        }
        return prev
      })

      // Play back stages 1–4 (Preprocess, Inference, Segment, Quant)
      startBackendStagePlayback()

      // Wait for stage playback (stages 0-5) to settle on 'done'
      const waitBackendDone = new Promise((resolve) => {
        const wanted = new Set(STAGES.slice(0, 5).map(s => s.id)) // Upload..Quant
        const tick = () => {
          setState(prev => {
            const allBackendDone = Array.from(wanted).every(id => prev.stages[id] === 'done')
            if (allBackendDone) resolve()
            else setTimeout(tick, 80)
            return prev
          })
        }
        const max = setTimeout(() => resolve(), 6000)
        tick()
        Promise.resolve().then(() => clearTimeout(max))
      })
      await waitBackendDone
      if (cancelledRef.current) return

      // Mark the 5 quantitative stages as 'done' in case of race
      setState(prev => {
        const stages = { ...prev.stages }
        for (const id of STAGES.slice(0, 5).map(s => s.id)) {
          if (stages[id] !== 'error') stages[id] = 'done'
        }
        return { ...prev, stages, overallProgress: recomputeOverallProgress(stages) }
      })

      // Store prediction result FIRST – ensures ResultDashboard can render
      // even if report generation fails
      const resultData = res.data
      setField({ result: resultData })

      // ── Stage 6: Actual PDF report generation ──────────────────
      const reportRes = await runReportGeneration(resultData)
      if (cancelledRef.current) return

      // Finalise (analysis + (report | report-error) complete)
      setState(prev => ({
        ...prev,
        status: 'done',
        overallProgress: 100,
      }))

      if (reportRes?.ok) {
        toast.success('Analysis complete · Report ready')
      } else if (reportRes?.cancelled) {
        // no-op
      } else {
        toast('Analysis complete — report generation failed', { icon: '⚠️' })
      }
    } catch (err) {
      clearTimeout(safetyTimer)
      cancelledRef.current = true
      clearTimers()
      const msg = err.message || 'Prediction failed'
      console.error('[usePrediction] Prediction pipeline failed:', msg)

      // Mark first non-done stage errored
      setState(prev => {
        const stages = { ...prev.stages }
        const order = STAGES.map(s => s.id)
        for (const id of order) {
          if (stages[id] !== 'done') { stages[id] = 'error'; break }
        }
        return {
          ...prev,
          stages,
          status: 'error',
          error: msg,
          overallProgress: recomputeOverallProgress(stages),
        }
      })
      toast.error(msg)
    }
  }, [
    state.imageFile, state.maskFile, state.studyMeta,
    setField, setStage, clearTimers, playBackendStages, runReportGeneration,
  ])

  const retryPrediction = useCallback(() => {
    runPrediction()
  }, [runPrediction])

  /* ── Public shape ──────────────────────────────────────────────────── */

  return {
    ...state,
    setImageFile,
    setMaskFile,
    setStudyMeta,
    runPrediction,
    retryPrediction,
    retryReport,
    reset,
    isLoading: state.status === 'uploading' || state.status === 'processing',
    isDone:    state.status === 'done',
    hasError:  state.status === 'error',
  }
}
