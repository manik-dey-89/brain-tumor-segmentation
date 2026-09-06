/**
 * API client for the Brain Tumor Segmentation backend.
 * All requests go to /api/v1 (proxied to localhost:8000 in dev,
 * or directly to VITE_API_URL in production).
 *
 * VITE_API_URL can be set to either:
 *   https://your-backend.onrender.com/api/v1   (full path - preferred)
 *   https://your-backend.onrender.com           (bare domain — normalised below)
 */
import axios from 'axios'

// Normalise: strip trailing slash, then ensure path ends with /api/v1
function _resolveBaseUrl() {
  const raw = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '')
  if (!raw) return '/api/v1'                         // dev: use Vite proxy
  if (raw.endsWith('/api/v1')) return raw            // already correct
  if (raw.endsWith('/api')) return `${raw}/v1`       // partial — append version
  return `${raw}/api/v1`                             // bare domain — append full prefix
}

const BASE_URL = _resolveBaseUrl()

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,   // 30s for normal requests (health, history, metrics)
})

// ── Helper: read a Blob as text / JSON ──────────────────────────────────────
async function blobToText(blob) {
  if (typeof blob.text === 'function') {
    try { return await blob.text() } catch {}
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result || '')
    reader.onerror = () => reject(reader.error || new Error('Blob read failed'))
    reader.readAsText(blob)
  })
}

async function extractBlobErrorDetail(blob) {
  try {
    const txt = await blobToText(blob)
    if (!txt) return null
    try {
      const parsed = JSON.parse(txt)
      return parsed?.detail || parsed?.message || parsed?.error || null
    } catch {
      return txt.length < 500 ? txt : txt.slice(0, 500)
    }
  } catch {
    return null
  }
}

// Response interceptor – normalise errors (handles both JSON and Blob responses)
api.interceptors.response.use(
  (res) => res,
  async (err) => {
    let msg = err.message || 'Unknown error'
    const data = err.response?.data
    if (data instanceof Blob) {
      const extracted = await extractBlobErrorDetail(data)
      if (extracted) msg = extracted
    } else if (data != null) {
      msg = data.detail || data.message || data.error || msg
    }
    const error = new Error(msg)
    if (err.response?.status != null) error.status = err.response.status
    return Promise.reject(error)
  }
)

// ── Endpoints ──────────────────────────────────────────────────────────────

export const checkHealth = () => api.get('/health')

/**
 * Ping the backend to wake it from Render free-tier sleep.
 * Call this before predict so cold-start latency is absorbed here,
 * not inside the 8-minute predict timeout.
 * Resolves regardless of outcome (never throws).
 */
export const wakeBackend = () =>
  api.get('/health', { timeout: 60_000 }).catch(() => {})

export const getModelInfo = () => api.get('/model-info')

/**
 * Upload MRI image (+ optional GT mask) and receive full prediction.
 * @param {File} imageFile
 * @param {File|null} maskFile   – optional ground-truth mask
 * @param {object|null} studyMeta – optional study/patient metadata object (StudyMeta)
 * @param {function} onProgress  – upload progress callback (0–100)
 */
export const predict = (imageFile, maskFile = null, studyMeta = null, onProgress = null) => {
  const form = new FormData()
  form.append('file', imageFile)
  if (maskFile) form.append('gt_mask', maskFile)
  if (studyMeta) {
    form.append('study_meta', JSON.stringify(studyMeta))
  }

  return api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 480_000,   // 8 min — CPU inference is slow on Render free tier
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 50))
      }
    },
  })
}

/**
 * Compose and download a PDF report for a completed prediction.
 * @param {object} payload – { prediction_result, images? }
 * @returns {Promise<Blob>} – PDF blob
 */
export const generateReportPdf = (payload) => {
  return api.post('/report/pdf', payload, {
    responseType: 'blob',
    timeout: 60_000,
  }).then((res) => res.data)
}

/**
 * Upload MRI and receive raw PNG mask bytes (Blob).
 */
export const segment = (imageFile) => {
  const form = new FormData()
  form.append('file', imageFile)
  return api.post('/segment', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  })
}

export const getMetrics = () => api.get('/metrics')

export const getHistory = (limit = 20, offset = 0) =>
  api.get('/history', { params: { limit, offset } })

export const getHistoryRecord = (id) => api.get(`/history/${id}`)

export const clearHistory = () => api.delete('/history')
