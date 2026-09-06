import React, { useState, useEffect } from 'react'
import {
  Upload, FileImage, Info, ChevronDown, ChevronUp,
  Lock, Eye, Target, ArrowLeft, ArrowRight,
} from 'lucide-react'
import UploadZone from './UploadZone'
import { fmtBytes, readImageDimensions } from '../utils/helpers'

function isNifti(name) {
  const n = name.toLowerCase()
  return n.endsWith('.nii') || n.endsWith('.nii.gz')
}

function MetaRow({ label, value }) {
  return (
    <div className="flex justify-between items-start py-2.5 border-b border-slate-100 last:border-0">
      <span className="text-xs text-slate-400 uppercase tracking-wider flex-shrink-0 w-28">{label}</span>
      <span className="text-sm font-medium text-slate-700 font-mono text-right truncate ml-3">
        {value ?? '—'}
      </span>
    </div>
  )
}

export default function UploadStep({
  enabled, studyMeta, imageFile, onImageFile,
  maskFile, onMaskFile, onStartAnalysis, onBack, uploadProgress,
}) {
  const [gtOpen,    setGtOpen]    = useState(false)
  const [imgDims,   setImgDims]   = useState(null)
  const [previewUrl,setPreviewUrl]= useState(null)

  useEffect(() => {
    if (!imageFile) { setImgDims(null); setPreviewUrl(null); return }
    setImgDims(null)
    readImageDimensions(imageFile).then(setImgDims)
    if (!isNifti(imageFile.name)) {
      const url = URL.createObjectURL(imageFile)
      setPreviewUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setPreviewUrl(null)
    }
  }, [imageFile])

  const canStart = enabled && imageFile != null

  /* ── Locked ── */
  if (!enabled) {
    return (
      <div className="relative glass-card p-8 sm:p-12">
        <div className="absolute inset-0 rounded-2xl bg-white/70 backdrop-blur-sm z-10
                        flex items-center justify-center">
          <div className="text-center space-y-3">
            <div className="w-14 h-14 mx-auto rounded-2xl bg-slate-100 border border-slate-200
                            flex items-center justify-center">
              <Lock className="w-6 h-6 text-slate-400" />
            </div>
            <h3 className="text-lg font-semibold text-slate-600">Upload Step Locked</h3>
            <p className="text-sm text-slate-400 max-w-sm">
              Complete and submit Patient Information (Step 1) before uploading an MRI scan.
            </p>
          </div>
        </div>
        <div className="opacity-30 grayscale pointer-events-none">
          <UploadZone onFile={() => {}} file={null} />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5">

      {/* Study badge */}
      {studyMeta?.study_id && (
        <div className="flex items-center gap-3 p-3 rounded-xl bg-medical-50 border border-medical-200">
          <div className="w-8 h-8 rounded-lg bg-medical-100 border border-medical-200
                          flex items-center justify-center flex-shrink-0">
            <Info className="w-4 h-4 text-medical-600" />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-0.5">Attached Study</p>
            <p className="text-sm font-mono font-semibold text-medical-700 tracking-wider truncate">
              {studyMeta.study_id} · {studyMeta.full_name} · {studyMeta.patient_id}
            </p>
          </div>
        </div>
      )}

      {/* Main upload card */}
      <div className="glass-card space-y-5">
        <div className="flex items-center gap-2">
          <Upload className="w-4 h-4 text-medical-600" />
          <h2 className="text-base font-semibold text-slate-800">Upload MRI Scan</h2>
        </div>

        <div className={imageFile ? '' : 'py-2'}>
          <UploadZone onFile={onImageFile} file={imageFile} label="MRI Scan" />
        </div>

        {/* Progress */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1.5">
              <span>Uploading…</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-200 overflow-hidden">
              <div className="h-full bg-medical-500 rounded-full transition-all duration-200"
                   style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}

        {/* File info + preview */}
        {imageFile && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">

            {/* Metadata */}
            <div className="rounded-xl bg-slate-50 border border-slate-200 p-4">
              <h3 className="flex items-center gap-2 text-xs font-semibold text-slate-500
                             uppercase tracking-wider mb-2">
                <FileImage className="w-3.5 h-3.5 text-medical-500" />
                Scan File Info
              </h3>
              <div className="divide-y divide-slate-100">
                <MetaRow label="Filename"  value={imageFile.name} />
                <MetaRow label="File Size" value={fmtBytes(imageFile.size)} />
                <MetaRow label="Type"      value={imageFile.type || 'application/octet-stream'} />
                <MetaRow
                  label="Dimensions"
                  value={imgDims?.volume
                    ? 'NIfTI – middle slice'
                    : imgDims?.width && imgDims?.height
                    ? `${imgDims.width} × ${imgDims.height} px`
                    : '—'}
                />
                <MetaRow
                  label="Format"
                  value={isNifti(imageFile.name)
                    ? 'NIfTI (Neuroimaging)'
                    : imageFile.name.split('.').pop().toUpperCase()}
                />
              </div>
            </div>

            {/* Preview */}
            <div className="rounded-xl bg-slate-50 border border-slate-200 p-4 flex flex-col">
              <h3 className="flex items-center gap-2 text-xs font-semibold text-slate-500
                             uppercase tracking-wider mb-3">
                <Eye className="w-3.5 h-3.5 text-medical-500" />
                Preview
              </h3>
              <div className="flex-1 min-h-[180px] rounded-lg overflow-hidden
                              bg-slate-100 border border-slate-200
                              flex items-center justify-center">
                {previewUrl ? (
                  <img src={previewUrl} alt="MRI preview"
                       className="w-full h-full object-contain max-h-[260px]" />
                ) : isNifti(imageFile.name) ? (
                  <div className="text-center p-6 space-y-2">
                    <div className="w-12 h-12 mx-auto rounded-xl bg-purple-100
                                    border border-purple-200 flex items-center justify-center">
                      <FileImage className="w-6 h-6 text-purple-500" />
                    </div>
                    <p className="text-sm font-medium text-slate-600">NIfTI Volume</p>
                    <p className="text-xs text-slate-400">
                      Middle axial slice extracted for 2-D segmentation.
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">Preview unavailable</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Research evaluation */}
      <div className="glass-card overflow-hidden p-0">
        <button
          onClick={() => setGtOpen((v) => !v)}
          className="w-full px-5 py-4 flex items-center justify-between text-left
                     hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-200
                            flex items-center justify-center">
              <Target className="w-4 h-4 text-indigo-500" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-700">Research Evaluation</p>
              <p className="text-xs text-slate-400">
                Optional: attach a ground-truth mask to compute Dice, IoU, and quality metrics
              </p>
            </div>
          </div>
          {gtOpen
            ? <ChevronUp   className="w-4 h-4 text-slate-400" />
            : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {gtOpen && (
          <div className="px-5 pb-5 pt-0 border-t border-slate-100 animate-fade-in">
            <div className="pt-4">
              <UploadZone onFile={onMaskFile} file={maskFile} label="Ground Truth Mask" compact />
              {maskFile && (
                <p className="text-xs text-slate-400 mt-3">
                  Evaluating against{' '}
                  <span className="text-slate-600 font-medium">{maskFile.name}</span>.
                  Metrics appear in the results dashboard after analysis.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between pt-1">
        <button onClick={onBack} className="btn-secondary flex items-center gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onStartAnalysis}
          disabled={!canStart}
          className="btn-primary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Upload className="w-4 h-4" />
          Start Analysis
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
