import React, { useState, useMemo } from 'react'
import {
  User, FileText, Stethoscope, CalendarDays, Hash,
  Mail, Phone, Building2, ClipboardList, CheckCircle2,
} from 'lucide-react'
import { generateStudyId } from '../utils/helpers'

const MODALITIES = [
  'T1-weighted',
  'T1-weighted with contrast',
  'T2-weighted',
  'FLAIR',
  'DWI',
  'ADC',
  'SWI',
  'Other',
]

const GENDERS = ['Male', 'Female', 'Other', 'Prefer not to say']

const todayStr = () => {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

const emptyForm = () => ({
  patientId: '', fullName: '', age: '', gender: '', contact: '',
  referringDoctor: '', hospital: '', scanDate: todayStr(),
  modality: '', clinicalNotes: '',
})

export default function PatientForm({ initialValues, onValid }) {
  const [form, setForm] = useState(() => {
    const base = emptyForm()
    if (initialValues) {
      return {
        ...base,
        patientId:       initialValues.patient_id       ?? initialValues.patientId       ?? base.patientId,
        fullName:        initialValues.full_name         ?? initialValues.fullName        ?? base.fullName,
        age:             initialValues.age != null ? String(initialValues.age) : base.age,
        gender:          initialValues.gender            ?? base.gender,
        contact:         initialValues.contact_email     ?? initialValues.contact         ?? base.contact,
        referringDoctor: initialValues.referring_doctor  ?? initialValues.referringDoctor ?? base.referringDoctor,
        hospital:        initialValues.hospital          ?? base.hospital,
        scanDate:        initialValues.scan_date         ?? initialValues.scanDate        ?? base.scanDate,
        modality:        initialValues.mri_modality      ?? initialValues.modality        ?? base.modality,
        clinicalNotes:   initialValues.clinical_notes    ?? initialValues.clinicalNotes   ?? base.clinicalNotes,
      }
    }
    return base
  })

  const [touched, setTouched] = useState(false)
  const [studyId, setStudyId] = useState(initialValues?.study_id || null)

  const update = (field) => (e) => setForm((p) => ({ ...p, [field]: e.target.value }))

  const errors = useMemo(() => {
    const e = {}
    if (!form.patientId.trim()) e.patientId = 'Patient ID is required'
    if (!form.fullName.trim())  e.fullName  = 'Full name is required'
    const ageNum = Number(form.age)
    if (form.age === '' || Number.isNaN(ageNum) || ageNum < 0 || ageNum > 130)
      e.age = 'Age must be 0 – 130'
    if (!form.gender)   e.gender   = 'Gender is required'
    if (!form.scanDate) e.scanDate = 'Scan date is required'
    if (!form.modality) e.modality = 'MRI modality is required'
    return e
  }, [form])

  const isValid = Object.keys(errors).length === 0

  const handleSubmit = (e) => {
    e.preventDefault()
    setTouched(true)
    if (!isValid) return
    const sid = studyId || generateStudyId()
    setStudyId(sid)
    onValid({
      study_id:         sid,
      patient_id:       form.patientId,
      full_name:        form.fullName,
      age:              Number(form.age),
      gender:           form.gender,
      contact_email:    form.contact        || null,
      referring_doctor: form.referringDoctor|| null,
      hospital:         form.hospital       || null,
      scan_date:        form.scanDate,
      mri_modality:     form.modality,
      clinical_notes:   form.clinicalNotes  || null,
    })
  }

  const showErr = (f) => touched && errors[f]

  /* ── Shared input class ── */
  const inputBase =
    'w-full bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm ' +
    'text-slate-800 placeholder-slate-400 ' +
    'focus:outline-none focus:ring-2 focus:ring-medical-400/40 focus:border-medical-400 ' +
    'transition-all shadow-sm'

  const row = (children) => (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">{children}</div>
  )

  const field = ({ label, icon: Icon, error, required, children, colSpan }) => (
    <div className={colSpan ? 'sm:col-span-2' : ''}>
      <label className="label-text flex items-center gap-1.5 mb-1.5">
        {Icon && <Icon className="w-3.5 h-3.5 text-slate-400" />}
        {label}
        {required && <span className="text-red-500">*</span>}
      </label>
      {children}
      {error && <p className="mt-1 text-xs text-red-500">{error}</p>}
    </div>
  )

  return (
    <form onSubmit={handleSubmit} className="glass-card space-y-5">
      <h2 className="section-title flex items-center gap-2">
        <User className="w-4 h-4 text-medical-600" />
        Patient Information
      </h2>

      {studyId && (
        <div className="flex items-center gap-3 p-3 rounded-xl
                        bg-medical-50 border border-medical-200">
          <CheckCircle2 className="w-5 h-5 text-medical-600 flex-shrink-0" />
          <div>
            <p className="label-text">Study ID</p>
            <p className="text-sm font-mono font-semibold text-medical-700 tracking-wider">
              {studyId}
            </p>
          </div>
        </div>
      )}

      {row(<>
        {field({
          label: 'Patient ID', icon: Hash, required: true,
          error: showErr('patientId') ? errors.patientId : null,
          children: (
            <input type="text" value={form.patientId} onChange={update('patientId')}
                   placeholder="e.g. MRN-12345" className={inputBase} />
          ),
        })}
        {field({
          label: 'Full Name', icon: User, required: true,
          error: showErr('fullName') ? errors.fullName : null,
          children: (
            <input type="text" value={form.fullName} onChange={update('fullName')}
                   placeholder="Jane Doe" className={inputBase} />
          ),
        })}
      </>)}

      {row(<>
        {field({
          label: 'Age', icon: CalendarDays, required: true,
          error: showErr('age') ? errors.age : null,
          children: (
            <input type="number" min="0" max="130" value={form.age} onChange={update('age')}
                   placeholder="e.g. 45" className={inputBase} />
          ),
        })}
        {field({
          label: 'Gender', icon: FileText, required: true,
          error: showErr('gender') ? errors.gender : null,
          children: (
            <select value={form.gender} onChange={update('gender')}
                    className={`${inputBase} appearance-none`}>
              <option value="" disabled>Select gender</option>
              {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          ),
        })}
      </>)}

      {row(<>
        {field({
          label: 'Contact / Email', icon: Mail,
          children: (
            <input type="text" value={form.contact} onChange={update('contact')}
                   placeholder="patient@example.com" className={inputBase} />
          ),
        })}
        {field({
          label: 'Referring Doctor', icon: Stethoscope,
          children: (
            <input type="text" value={form.referringDoctor} onChange={update('referringDoctor')}
                   placeholder="Dr. Smith" className={inputBase} />
          ),
        })}
      </>)}

      {row(<>
        {field({
          label: 'Hospital / Clinic', icon: Building2,
          children: (
            <input type="text" value={form.hospital} onChange={update('hospital')}
                   placeholder="City General Hospital" className={inputBase} />
          ),
        })}
        {field({
          label: 'Scan Date', icon: CalendarDays, required: true,
          error: showErr('scanDate') ? errors.scanDate : null,
          children: (
            <input type="date" value={form.scanDate} onChange={update('scanDate')}
                   className={inputBase} />
          ),
        })}
      </>)}

      {row(field({
        label: 'MRI Modality', icon: ClipboardList, required: true,
        error: showErr('modality') ? errors.modality : null,
        children: (
          <select value={form.modality} onChange={update('modality')}
                  className={`${inputBase} appearance-none`}>
            <option value="" disabled>Select MRI sequence</option>
            {MODALITIES.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ),
      }))}

      {row(field({
        label: 'Clinical Notes', icon: FileText, colSpan: true,
        children: (
          <textarea value={form.clinicalNotes} onChange={update('clinicalNotes')}
                    rows={4} placeholder="Relevant history, symptoms, indications…"
                    className={`${inputBase} resize-y min-h-[96px]`} />
        ),
      }))}

      <div className="pt-2 flex justify-end">
        <button type="submit" className="btn-primary flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5" />
          {studyId ? 'Update & Continue' : 'Confirm & Continue'}
        </button>
      </div>
    </form>
  )
}
