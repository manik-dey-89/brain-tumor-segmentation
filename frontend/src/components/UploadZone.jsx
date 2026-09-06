import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { ImageIcon, X, FileImage, CloudUpload } from 'lucide-react'

const ACCEPTED = {
  'image/png':  ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/tiff': ['.tif', '.tiff'],
  'application/octet-stream': ['.nii', '.nii.gz'],
}

export default function UploadZone({ onFile, file, label = 'MRI Image', compact = false }) {
  const onDrop = useCallback((accepted) => {
    if (accepted[0]) onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
  })

  /* Border / bg state */
  const borderCls = isDragReject
    ? 'border-red-400 bg-red-50'
    : isDragActive
    ? 'border-medical-400 bg-medical-50 scale-[1.01]'
    : file
    ? 'border-emerald-400 bg-emerald-50'
    : 'border-slate-300 hover:border-medical-400 hover:bg-medical-50/50'

  /* ── Compact chip when file loaded ── */
  if (compact && file) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl bg-emerald-50 border border-emerald-200">
        <FileImage className="w-5 h-5 text-emerald-600 flex-shrink-0" />
        <span className="text-sm text-emerald-700 truncate flex-1">{file.name}</span>
        <button
          onClick={(e) => { e.stopPropagation(); onFile(null) }}
          className="p-1 rounded-lg hover:bg-emerald-100 transition-colors"
        >
          <X className="w-4 h-4 text-emerald-500" />
        </button>
      </div>
    )
  }

  return (
    <div
      {...getRootProps()}
      className={`relative border-2 border-dashed rounded-2xl transition-all duration-200
                  cursor-pointer select-none bg-white ${borderCls}
                  ${compact ? 'p-4' : 'p-6 sm:p-10'}`}
    >
      <input {...getInputProps()} />

      {isDragActive && !compact && (
        <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
          <div className="scan-line h-8 w-full absolute top-0" />
        </div>
      )}

      {file ? (
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-emerald-100 border border-emerald-200
                          flex items-center justify-center flex-shrink-0">
            <ImageIcon className="w-6 h-6 text-emerald-600" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-emerald-700 truncate">{file.name}</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {(file.size / 1024).toFixed(1)} KB · Click to change
            </p>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onFile(null) }}
            className="p-2 rounded-xl hover:bg-slate-100 transition-colors flex-shrink-0"
          >
            <X className="w-4 h-4 text-slate-400" />
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center text-center gap-4">
          <div className={`rounded-2xl flex items-center justify-center transition-all duration-200
                          ${isDragActive
                            ? 'bg-medical-100 border border-medical-300 scale-110'
                            : 'bg-medical-50 border border-medical-200'}
                          ${compact ? 'w-10 h-10' : 'w-16 h-16'}`}>
            <CloudUpload className={`text-medical-500
                                     ${compact ? 'w-5 h-5' : 'w-8 h-8'}
                                     ${isDragActive ? 'animate-bounce' : ''}`} />
          </div>

          {!compact && (
            <>
              <div className="space-y-1">
                <p className="text-base font-semibold text-slate-700">
                  {isDragActive ? 'Drop to upload' : `Drag & drop your ${label}`}
                </p>
                <p className="text-sm text-slate-400">
                  or{' '}
                  <span className="text-medical-600 underline underline-offset-2">
                    click to browse
                  </span>
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                {['PNG','JPG','TIFF','NIfTI'].map((f) => (
                  <span key={f} className="px-2 py-0.5 rounded-md bg-slate-100 border border-slate-200">
                    {f}
                  </span>
                ))}
                <span className="text-slate-300">· Max 50 MB</span>
              </div>
            </>
          )}

          {compact && (
            <p className="text-xs text-slate-400">
              {isDragActive ? 'Drop here' : `Add ${label}`}
            </p>
          )}

          {isDragReject && (
            <p className="text-xs text-red-500 font-medium">File type not supported</p>
          )}
        </div>
      )}
    </div>
  )
}
