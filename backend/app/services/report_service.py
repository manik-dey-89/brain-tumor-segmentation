"""
PDF Report Service
==================
Generates professional multi-section radiology-style research reports
from Brain Tumor Segmentation prediction results using ReportLab.

Produces A4 portrait PDF with: header, patient/study table, scan info,
MRI image + overlay, quantitative findings, GT metrics (if present),
methodology, limitations, and a formal disclaimer.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image as _PILImage  # noqa: F401 – kept alive for safety
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────

NEURO_DARK   = colors.HexColor("#0f172a")
NEURO_MED    = colors.HexColor("#1e293b")
ACCENT       = colors.HexColor("#0ea5e9")   # medical-500
ACCENT_DARK  = colors.HexColor("#0369a1")
RULE         = colors.HexColor("#334155")
BAND_EVEN    = colors.HexColor("#f8fafc")
TEXT_DIM     = colors.HexColor("#64748b")
TEXT_BODY    = colors.HexColor("#1e293b")
DISCLAIM_BG  = colors.HexColor("#fff7ed")
DISCLAIM_BD  = colors.HexColor("#fb923c")
POSITIVE_BG  = colors.HexColor("#fef2f2")
POSITIVE_BD  = colors.HexColor("#ef4444")
NEGATIVE_BG  = colors.HexColor("#ecfdf5")
NEGATIVE_BD  = colors.HexColor("#10b981")

# ── Typography helpers ───────────────────────────────────────────────────────

def _styles():
    sheet = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sheet["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NEURO_DARK,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sheet["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=ACCENT_DARK,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sheet["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NEURO_DARK,
            spaceBefore=14,
            spaceAfter=6,
            borderPadding=0,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=sheet["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=ACCENT_DARK,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=TEXT_BODY,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "body_sm": ParagraphStyle(
            "BodySm",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=TEXT_DIM,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sheet["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=TEXT_DIM,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=sheet["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#9a3412"),
            alignment=TA_JUSTIFY,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=sheet["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=TEXT_DIM,
        ),
    }
    return styles


# ── Data format helpers ───────────────────────────────────────────────────────

def _fmt(v, decimals=4, default="—"):
    if v is None:
        return default
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v, decimals=2):
    return _fmt(v, decimals=decimals) + "%"


def _fmt_ms(ms):
    if ms is None:
        return "—"
    ms = float(ms)
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def _fmt_date(iso):
    if not iso:
        return "Not provided"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y  %H:%M UTC")
    except Exception:
        return str(iso)


def _decode_b64(data_uri: str) -> bytes:
    """Strip data URI prefix and decode base64 to bytes."""
    if not data_uri:
        return b""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    try:
        return base64.b64decode(data_uri)
    except Exception:
        return b""


def _b64_to_image_flowable(data_uri: str, max_w_cm: float, max_h_cm: float):
    """Convert a base64 data-URI to a ReportLab Image flowable.

    Passes the raw image *bytes* to ReportLab so the library copies them,
    avoiding any issue with BytesIO lifetime / garbage-collection before build().
    """
    raw = _decode_b64(data_uri)
    if not raw:
        return None
    if not _HAS_PIL:
        logger.warning("Pillow (PIL) not installed – skipping PDF image")
        return None
    try:
        with io.BytesIO(raw) as probe:
            with _PILImage.open(probe) as im:
                w_px, h_px = im.size
                aspect = w_px / max(h_px, 1)
                im.load()
        max_w = max_w_cm * cm
        max_h = max_h_cm * cm
        w = min(max_w, aspect * max_h)
        h = w / aspect
        if h > max_h:
            h = max_h
            w = h * aspect
        # Pass a COPY of the bytes – ReportLab will own the reference
        return Image(io.BytesIO(raw), width=w, height=h)
    except Exception as exc:
        logger.warning("Could not embed image in PDF: %s", exc)
        return None


# ── Table helpers ─────────────────────────────────────────────────────────────

def _kv_table(rows, col_widths=None):
    """Left-label / right-value info table."""
    data = [["Field", "Value"]] + [[Paragraph(k, _styles()["body_sm"]), Paragraph(str(v), _styles()["body"])] for k, v in rows]
    t = Table(data, colWidths=col_widths or [4.5 * cm, 12 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), ACCENT_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",  (0, 0), (-1, 0), 6),
        ("ALIGN",       (0, 0), (-1, 0), "LEFT"),
        ("GRID",        (0, 0), (-1, -1), 0.3, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND_EVEN]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ]))
    return t


def _metrics_table(rows, header, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), ACCENT_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING",  (0, 0), (-1, 0), 5),
        ("GRID",        (0, 0), (-1, -1), 0.3, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND_EVEN]),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def _disclaimer_box(text):
    """Boxed disclaimer paragraph."""
    inner = [[Paragraph(text, _styles()["disclaimer"])]]
    t = Table(inner, colWidths=[16.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DISCLAIM_BG),
        ("BOX",        (0, 0), (-1, -1), 1.2, DISCLAIM_BD),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _status_band(tumor_detected, confidence):
    s = _styles()
    txt = "No Tumour Region Detected" if not tumor_detected else "Tumour Region Detected — AI-Assisted Research Finding"
    bg  = NEGATIVE_BG if not tumor_detected else POSITIVE_BG
    bd  = NEGATIVE_BD if not tumor_detected else POSITIVE_BD
    conf = _fmt_pct(confidence * 100 if confidence is not None and confidence <= 1 else confidence, 1)
    conf_txt = f"Model confidence: {conf}"
    t = Table([[Paragraph(f"<b>{txt}</b>", s["h3"]), Paragraph(conf_txt, s["body_sm"])]],
              colWidths=[10.5 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.8, bd),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (1, 0), (1, 0), "RIGHT"),
    ]))
    return t


# ── Page decoration (header rule + footer) ───────────────────────────────────

class _DecoratedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height,
            id="normal",
        )
        template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._draw_decorations,
        )
        self.addPageTemplates([template])

    @staticmethod
    def _draw_decorations(canv, doc):
        canv.saveState()
        w, h = A4
        # Top rule
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(1.4)
        canv.line(doc.leftMargin, h - doc.topMargin + 4 * mm,
                  w - doc.rightMargin, h - doc.topMargin + 4 * mm)
        # Header title (right aligned)
        canv.setFont("Helvetica-Bold", 8)
        canv.setFillColor(ACCENT_DARK)
        canv.drawRightString(w - doc.rightMargin, h - doc.topMargin + 8 * mm,
                             "Brain Tumor Segmentation — AI-Assisted Research Report")
        # Footer rule
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, doc.bottomMargin - 6 * mm,
                  w - doc.rightMargin, doc.bottomMargin - 6 * mm)
        # Footer centre – page number
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(TEXT_DIM)
        canv.drawCentredString(w / 2.0, doc.bottomMargin - 10 * mm,
                               f"Page {doc.page}")
        # Footer left – disclaimer line
        canv.setFont("Helvetica", 7)
        canv.drawString(doc.leftMargin, doc.bottomMargin - 10 * mm,
                        "Research output — NOT a medical diagnosis.")
        canv.restoreState()


# ── Main build function ───────────────────────────────────────────────────────

def build_report_pdf(prediction_result: dict[str, Any], images: Optional[dict[str, str]] = None) -> bytes:
    """
    Build a PDF report from a Brain Tumor Segmentation prediction result.

    Parameters
    ----------
    prediction_result : dict – the payload returned by /predict
    images            : optional dict of name -> base64 data-URI to use
                        (falls back to prediction_result['images'] if omitted)
    Returns
    -------
    bytes – raw PDF file content
    """
    s = _styles()
    imgs = images or prediction_result.get("images") or {}

    study   = prediction_result.get("study") or {}
    model   = prediction_result.get("model_info") or {}
    metrics = prediction_result.get("metrics") or {}
    det     = bool(prediction_result.get("tumor_detected"))
    conf    = prediction_result.get("confidence")
    tumour_pct = prediction_result.get("tumor_percentage")
    tumour_px  = prediction_result.get("tumor_pixels")
    total_px   = prediction_result.get("total_pixels")
    inf_ms     = prediction_result.get("inference_ms")
    total_ms   = prediction_result.get("total_ms")
    filename   = prediction_result.get("filename", "—")
    analysed   = prediction_result.get("timestamp")
    study_id   = study.get("study_id", prediction_result.get("prediction_id", "UNKNOWN"))

    # Build in-memory PDF
    buf = io.BytesIO()
    doc = _DecoratedDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=f"Brain Tumor Segmentation Report — {study_id}",
        author="Brain Tumor Segmentation Research Platform",
    )

    story = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. REPORT HEADER
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Brain Tumor Segmentation", s["title"]))
    story.append(Paragraph("AI-Assisted Brain Tumour Segmentation — Research Report", s["subtitle"]))

    header_info = [
        ["Study / Case ID",      f"<b>{study_id}</b>"],
        ["Report Generated",     _fmt_date(datetime.utcnow().isoformat() + "Z")],
        ["Analysis Timestamp",   _fmt_date(analysed)],
        ["Prediction ID",        prediction_result.get("prediction_id", "—")],
    ]
    story.append(_kv_table(header_info, col_widths=[4.5 * cm, 12 * cm]))
    story.append(Spacer(1, 6))
    story.append(_status_band(det, conf))

    # ═══════════════════════════════════════════════════════════════════════
    # 2. PATIENT & STUDY DETAILS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Patient &amp; Study Details", s["h2"]))
    patient_rows = [
        ["Patient ID",        study.get("patient_id", "Not provided")],
        ["Full Name",         study.get("full_name", "Not provided")],
        ["Age",               study.get("age", "Not provided")],
        ["Gender",            study.get("gender", "Not provided")],
        ["Contact / Email",   study.get("contact_email") or study.get("contact") or "Not provided"],
        ["Referring Doctor",  study.get("referring_doctor", "Not provided")],
        ["Hospital / Clinic", study.get("hospital", "Not provided")],
        ["Scan Date",         study.get("scan_date", "Not provided")],
        ["MRI Modality",      study.get("mri_modality") or study.get("modality") or "Not provided"],
        ["Clinical Notes",    study.get("clinical_notes") or study.get("clinicalNotes") or "—"],
    ]
    story.append(_kv_table(patient_rows, col_widths=[4.5 * cm, 12 * cm]))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SCAN INFORMATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Scan Information", s["h2"]))
    scan_rows = [
        ["Original Filename",     filename],
        ["MRI Modality",          study.get("mri_modality") or study.get("modality") or "Not provided"],
        ["Processed Size",        "256 × 256 px (Attention U-Net input)"],
        ["Total Scan Pixels",     f"{int(total_px):,}" if total_px else "—"],
        ["File Format",           Path(filename).suffix.lstrip(".").upper() or "Unknown"],
        ["Ground Truth Mask",     "Provided" if metrics and any(v is not None for k, v in metrics.items() if k in ("dice", "iou")) else "Not provided"],
    ]
    story.append(_kv_table(scan_rows, col_widths=[4.5 * cm, 12 * cm]))

    # ═══════════════════════════════════════════════════════════════════════
    # 4 & 5. MRI IMAGE + SEGMENTATION OVERLAY
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. MRI Image &amp; Predicted Segmentation", s["h2"]))

    orig    = _b64_to_image_flowable(imgs.get("original"), 7.8, 7.8)
    overlay = _b64_to_image_flowable(imgs.get("overlay"),  7.8, 7.8)

    if orig and overlay:
        image_table = Table(
            [[orig or "", overlay or ""]],
            colWidths=[8.0 * cm, 8.0 * cm],
        )
        image_table.setStyle(TableStyle([
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",       (0, 0), (-1, -1), 0.4, RULE),
            ("BACKGROUND", (0, 0), (-1, -1), colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(image_table)
    elif orig:
        story.append(orig)
    elif overlay:
        story.append(overlay)

    if orig:
        story.append(Paragraph("Fig 1. Processed MRI slice input (windowed for display).", s["caption"]))
    if overlay:
        story.append(Paragraph(
            "Fig 2. Segmentation overlay — red / orange region indicates AI-predicted tumour candidate pixels.",
            s["caption"],
        ))

    # ═══════════════════════════════════════════════════════════════════════
    # 6. QUANTITATIVE FINDINGS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Quantitative Findings", s["h2"]))
    q_rows = [
        ["Tumour Region Detected",  "Yes" if det else "No"],
        ["Tumour Area Percentage",  _fmt_pct(tumour_pct, 2)],
        ["Tumour Pixel Count",      f"{int(tumour_px):,}" if tumour_px is not None else "—"],
        ["Total Scan Pixels",       f"{int(total_px):,}"  if total_px  is not None else "—"],
        ["Model Confidence",        _fmt_pct(conf * 100 if conf is not None and conf <= 1 else conf, 1)],
        ["Model Inference Time",    _fmt_ms(inf_ms)],
        ["Total Pipeline Time",     _fmt_ms(total_ms)],
    ]
    story.append(_kv_table(q_rows, col_widths=[5 * cm, 11.5 * cm]))

    # ═══════════════════════════════════════════════════════════════════════
    # 7. GROUND TRUTH EVALUATION (conditional)
    # ═══════════════════════════════════════════════════════════════════════
    gt_keys = [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1"),
               ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
    gt_values_present = [metrics.get(k) for k, _ in gt_keys if metrics.get(k) is not None]

    if gt_values_present:
        story.append(Paragraph("5. Ground Truth Evaluation", s["h2"]))
        story.append(Paragraph(
            "Metrics computed against the user-provided ground-truth mask.",
            s["body_sm"],
        ))
        story.append(Spacer(1, 4))
        header = ["Metric", "Value", "Metric", "Value"]
        # Pack key/value pairs into 2-col wide table
        packed = []
        pairs = [(lbl, _fmt(metrics.get(k), 4)) for k, lbl in gt_keys]
        for i in range(0, len(pairs), 2):
            a = pairs[i]
            b = pairs[i + 1] if i + 1 < len(pairs) else ("", "")
            packed.append([a[0], a[1], b[0], b[1]])
        story.append(_metrics_table(packed, header, col_widths=[3.8 * cm, 3.0 * cm, 3.8 * cm, 3.0 * cm]))

    # ═══════════════════════════════════════════════════════════════════════
    # 8. METHODOLOGY (short paragraph)
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("6. Methodology", s["h2"]))
    methodology = (
        "The uploaded MRI is first resized to 256 × 256, converted to single-channel "
        "grayscale, intensity-normalised using per-slice min–max scaling, and "
        "denoised with a mild Gaussian blur. The processed tensor is fed through "
        "an Attention U-Net convolutional network (encoder-decoder with skip "
        "connections and attention gating) trained on synthetic brain tumour MRI. "
        "If test-time augmentation (TTA) is enabled, the model outputs are averaged "
        "over geometric transforms. The resulting probability map is thresholded "
        "at the configured cut-off, then post-processed with morphological "
        "connected-component filtering and hole-filling. Tumour area and confidence "
        "(mean predicted probability inside the segmented region) are derived "
        "directly from the probability map."
    )
    story.append(Paragraph(methodology, s["body"]))

    # ═══════════════════════════════════════════════════════════════════════
    # 9. LIMITATIONS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("7. Limitations &amp; Review", s["h2"]))
    limitations = [
        "Operates on a single 2D axial slice; not a full multi-parametric volumetric study.",
        "Trained on synthetic MRI data; performance on real clinical acquisitions may vary.",
        "Sensitivity to MRI acquisition protocol, field strength, coil profile, and reconstruction kernel.",
        "Ground-truth annotations and radiologist review are required before any clinical use.",
        "Does not substitute for a complete neuroradiological examination or provide tumour grading / staging.",
    ]
    bullets = "<br/>".join(f"• {line}" for line in limitations)
    story.append(Paragraph(bullets, s["body"]))

    # ═══════════════════════════════════════════════════════════════════════
    # 10. FORMAL DISCLAIMER (boxed)
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 6))
    story.append(_disclaimer_box(
        "AI-assisted research output — NOT a medical diagnosis. For research and "
        "educational use only. Results must be reviewed by a qualified radiologist "
        "prior to any clinical decision. Brain Tumor Segmentation does not replace standard of care "
        "imaging review or specialist consultation."
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
