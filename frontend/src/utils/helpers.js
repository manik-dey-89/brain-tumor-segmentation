/** Format a number as a percentage string */
export const fmtPct = (v, decimals = 2) =>
  v != null ? `${Number(v).toFixed(decimals)}%` : '—'

/** Format a metric value (0-1 range) */
export const fmtMetric = (v, decimals = 4) =>
  v != null ? Number(v).toFixed(decimals) : '—'

/** Format milliseconds to a human string */
export const fmtMs = (ms) => {
  if (ms == null) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(1)} s`
}

/** Format large numbers with commas */
export const fmtNum = (n) =>
  n != null ? Number(n).toLocaleString() : '—'

/** Colour class for a metric value (higher = better) */
export const metricColour = (v) => {
  if (v == null) return 'text-slate-300'
  if (v >= 0.8)  return 'text-emerald-600'
  if (v >= 0.6)  return 'text-amber-500'
  return 'text-red-500'
}

/** Confidence badge colour */
export const confidenceColour = (v) => {
  if (v == null) return 'bg-slate-100 text-slate-400 border-slate-200'
  if (v >= 0.7)  return 'bg-emerald-50 text-emerald-700 border-emerald-300'
  if (v >= 0.5)  return 'bg-amber-50   text-amber-700   border-amber-300'
  return 'bg-red-50 text-red-700 border-red-300'
}

/** Format ISO timestamp to readable string */
export const fmtDate = (iso) => {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return iso
  }
}

/** Download a blob as a file */
export const downloadBlob = (blob, filename) => {
  const url  = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href     = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/** Download a base64 data-URI as a file */
export const downloadBase64 = (dataUri, filename) => {
  const link = document.createElement('a')
  link.href     = dataUri
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/** Convert bytes to human-readable size */
export const fmtBytes = (bytes) => {
  if (bytes == null) return '—'
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 ** 2)   return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

/** Generate a unique study ID: NSS-YYYYMMDD-NNNN (4 decimal digits, matches backend pattern ^NSS-\d{8}-\d{4}$) */
export const generateStudyId = () => {
  const date = new Date()
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  const digits = String(Math.floor(Math.random() * 10000)).padStart(4, '0')
  return `NSS-${y}${m}${d}-${digits}`
}

/** Read image dimensions; for .nii/.nii.gz return volume:true */
export const readImageDimensions = (file) => {
  return new Promise((resolve) => {
    const name = file.name.toLowerCase()
    if (name.endsWith('.nii') || name.endsWith('.nii.gz')) {
      resolve({ width: null, height: null, volume: true })
      return
    }
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      const width = img.naturalWidth
      const height = img.naturalHeight
      URL.revokeObjectURL(url)
      resolve({ width, height })
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      resolve({ width: null, height: null })
    }
    img.src = url
  })
}
