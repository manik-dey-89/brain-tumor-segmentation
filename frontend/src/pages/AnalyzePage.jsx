import React, { useState, useEffect } from 'react'
import { Brain, User, Upload, Activity, CheckCircle2, Lock } from 'lucide-react'
import PatientForm from '../components/PatientForm'
import UploadStep from '../components/UploadStep'
import AnalysisPipeline from '../components/AnalysisPipeline'
import ResultDashboard from '../components/ResultDashboard'
import usePrediction from '../hooks/usePrediction'
import ErrorBoundary from '../components/ErrorBoundary'

const STEPS = [
  { id: 1, label: 'Patient Information', icon: User },
  { id: 2, label: 'MRI Upload',          icon: Upload },
  { id: 3, label: 'Analysis',            icon: Activity },
]

function Stepper({ activeStep, step1Valid }) {
  return (
    <div className="glass-card p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2 sm:gap-4">
        {STEPS.map((step, idx) => {
          const Icon = step.icon
          const isActive = activeStep === step.id
          const isPast   = activeStep > step.id
          const isLocked = step.id > 1 && !step1Valid && !isPast
          return (
            <React.Fragment key={step.id}>
              <div className={`flex items-center gap-2 sm:gap-3 flex-1 min-w-0
                               ${isLocked ? 'opacity-40 grayscale' : ''}`}>
                <div className={`w-9 h-9 sm:w-10 sm:h-10 flex-shrink-0 rounded-xl
                                 flex items-center justify-center border
                                 transition-all duration-200 ${
                  isActive
                    ? 'bg-medical-600 border-medical-500 text-white shadow-md shadow-medical-200'
                    : isPast
                    ? 'bg-emerald-50 border-emerald-300 text-emerald-600'
                    : 'bg-slate-100 border-slate-200 text-slate-400'
                }`}>
                  {isPast   ? <CheckCircle2 className="w-5 h-5" />
                  : isLocked ? <Lock className="w-4 h-4" />
                  : <span className="text-sm font-bold">{step.id}</span>}
                </div>
                <div className="min-w-0">
                  <p className={`text-xs sm:text-sm font-semibold truncate ${
                    isActive ? 'text-slate-900' : isPast ? 'text-slate-600' : 'text-slate-400'
                  }`}>
                    Step {step.id}
                  </p>
                  <p className="hidden sm:block text-xs text-slate-400 truncate">{step.label}</p>
                </div>
              </div>
              {idx < STEPS.length - 1 && (
                <div className={`hidden sm:block flex-1 h-px rounded-full transition-colors duration-300 ${
                  isPast || idx < activeStep - 1 ? 'bg-medical-300' : 'bg-slate-200'
                }`} />
              )}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}

export default function AnalyzePage() {
  const pred = usePrediction()

  const [activeStep,  setActiveStep]  = useState(1)
  const [studyMeta,   setStudyMeta]   = useState(null)
  const [step1Valid,  setStep1Valid]  = useState(false)

  useEffect(() => {
    if (studyMeta) pred.setStudyMeta(studyMeta)
  }, [studyMeta, pred])

  const showResult = pred.isDone && pred.result

  const handleStep1Valid = (meta) => {
    setStudyMeta(meta)
    setStep1Valid(true)
    setActiveStep(2)
  }

  const handleStartAnalysis = () => {
    setActiveStep(3)
    pred.runPrediction()
  }

  const handleBackFromStep2 = () => setActiveStep(1)
  const handleBackFromStep3 = () => setActiveStep(2)

  const handleNewAnalysis = () => {
    pred.reset()
    setActiveStep(1)
    setStudyMeta(null)
    setStep1Valid(false)
  }

  const showPipelineHud      = !showResult && activeStep === 3
  const showReportErrorPanel = showResult && pred.pdfError && !pred.error

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 space-y-6">

      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-900">
          <Brain className="w-8 h-8 text-medical-600" />
          Brain Tumour Analysis
        </h1>
        <p className="text-slate-500 mt-2">
          Complete the 3-step workflow to register a study case, upload an MRI scan,
          and run Attention U-Net tumour segmentation.
        </p>
      </div>

      {!showResult && <Stepper activeStep={activeStep} step1Valid={step1Valid} />}

      <ErrorBoundary
        title="Analyze Page Render Error"
        message="Something crashed while running the tumour analysis workflow."
        onReset={handleNewAnalysis}
      >
        <div className="animate-fade-in space-y-5">
          {!showResult && activeStep === 1 && (
            <PatientForm initialValues={studyMeta || undefined} onValid={handleStep1Valid} />
          )}

          {!showResult && activeStep === 2 && (
            <UploadStep
              enabled={step1Valid}
              studyMeta={studyMeta}
              imageFile={pred.imageFile}
              onImageFile={pred.setImageFile}
              maskFile={pred.maskFile}
              onMaskFile={pred.setMaskFile}
              onStartAnalysis={handleStartAnalysis}
              onBack={handleBackFromStep2}
              uploadProgress={pred.uploadProgress}
            />
          )}

          {showPipelineHud && (
            <AnalysisPipeline
              stageStates={pred.stages}
              errorMsg={pred.error}
              reportError={pred.pdfError}
              startTime={pred.startTime}
              overallProgress={pred.overallProgress}
              onRetry={pred.retryPrediction}
              onRetryReport={pred.retryReport}
              onBack={handleBackFromStep3}
            />
          )}

          {showReportErrorPanel && (
            <AnalysisPipeline
              stageStates={pred.stages}
              reportError={pred.pdfError}
              overallProgress={pred.overallProgress}
              onRetryReport={pred.retryReport}
              onBack={handleBackFromStep3}
            />
          )}

          {showResult && (
            <ResultDashboard
              result={pred.result}
              pdfBlob={pred.pdfBlob}
              pdfError={pred.pdfError}
              onRetryReport={pred.retryReport}
              onNewAnalysis={handleNewAnalysis}
            />
          )}
        </div>
      </ErrorBoundary>
    </div>
  )
}
