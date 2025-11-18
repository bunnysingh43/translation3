import streamlit as st
import time
import io
from datetime import datetime

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Assamese PDF OCR & Translation",
    layout="wide",
    page_icon="🔤"
)

st.title("🔤 Assamese PDF OCR & Translation System")
st.markdown("**100% Free OCR | Assamese → English | Layout Preserved**")

# -----------------------------
# LAZY IMPORT HELPERS (IMPORTANT)
# -----------------------------
def lazy_import_fitz():
    try:
        import fitz
        return fitz, None
    except Exception as e:
        return None, str(e)

def lazy_import_pytesseract():
    try:
        import pytesseract
        return pytesseract, None
    except Exception as e:
        return None, str(e)

def lazy_import_PIL():
    try:
        from PIL import Image, ImageEnhance
        return Image, ImageEnhance, None
    except Exception as e:
        return None, None, str(e)

def lazy_import_translator():
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator, None
    except Exception as e:
        return None, str(e)

def lazy_import_reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib import colors
        from xml.sax.saxutils import escape
        return {
            "A4": A4,
            "getSampleStyleSheet": getSampleStyleSheet,
            "ParagraphStyle": ParagraphStyle,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Table": Table,
            "TableStyle": TableStyle,
            "Paragraph": Paragraph,
            "Spacer": Spacer,
            "PageBreak": PageBreak,
            "inch": inch,
            "TA_LEFT": TA_LEFT,
            "TA_CENTER": TA_CENTER,
            "colors": colors,
            "escape": escape
        }, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Features:**
    - 🔤 Tesseract OCR with Assamese language
    - 🌐 Free Google Translate (no API key)
    - ⏱️ Intelligent rate limiting
    - 🔄 Automatic retry with backoff
    - 📐 Layout preservation
    - 📊 Real-time progress
    """)

    st.markdown("### ⚙️ Settings")
    delay_between_pages = st.slider("Delay between pages (seconds)", 0.5, 5.0, 1.5, 0.5)
    max_retries = st.number_input("Max retries per translation", 1, 10, 3)
    dpi = st.number_input("OCR DPI Quality", 150, 600, 300, 50)

# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
def preprocess_image(image):
    _, _, err = lazy_import_PIL()
    if err:
        st.error("PIL import error: " + err)
        return image

    Image, ImageEnhance, _ = lazy_import_PIL()
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(2.0)

# -----------------------------
# TRANSLATION (LAZY)
# -----------------------------
def translate_text_with_retry(text, max_retries=3):
    if not text or text.strip() == "":
        return ""

    GoogleTranslator, err = lazy_import_translator()
    if err:
        st.error("Translation module import failed: " + err)
        return text

    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)

        except Exception as e:
            if attempt == max_retries - 1:
                return f"[FAILED: {text[:50]}]"

            time.sleep(1 * (attempt + 1))

    return text

# -----------------------------
# TABLE PARSING HELPERS (same logic)
# -----------------------------
# (Shortened only for structure — your logic is unchanged)
# I am keeping all your functions exactly as-is:

def detect_column_centers(words_data, expected_columns=7):
    if not words_data:
        return []
    x_positions = sorted([w['left'] + w['width'] / 2 for w in words_data])
    if len(x_positions) < 2:
        return x_positions if x_positions else []
    min_x = min(x_positions)
    max_x = max(x_positions)
    all_gaps = []
    for i in range(len(x_positions) - 1):
        gap_size = x_positions[i + 1] - x_positions[i]
        all_gaps.append((x_positions[i], x_positions[i + 1], gap_size))
    if not all_gaps:
        equal_width = (max_x - min_x) / expected_columns
        return [min_x + equal_width * (i + 0.5) for i in range(expected_columns)]
    avg_gap = sum([g[2] for g in all_gaps]) / len(all_gaps)
    median_gap = sorted([g[2] for g in all_gaps])[len(all_gaps) // 2]
    threshold = max(median_gap * 2, avg_gap * 1.5, 30)
    significant_gaps = [g for g in all_gaps if g[2] >= threshold]
    if len(significant_gaps) < expected_columns - 1:
        significant_gaps = sorted(all_gaps, key=lambda x: x[2], reverse=True)[:expected_columns - 1]
    split_points = sorted([g[0] + g[2] / 2 for g in significant_gaps])[:expected_columns - 1]
    boundaries = [min_x] + split_points + [max_x]
    column_centers = []
    for i in range(len(boundaries) - 1):
        in_range = [x for x in x_positions if boundaries[i] <= x < boundaries[i + 1]]
        if in_range:
            column_centers.append(sum(in_range) / len(in_range))
        else:
            center = (boundaries[i] + boundaries[i + 1]) / 2
            column_centers.append(center)
    return column_centers[:expected_columns]

def cluster_rows_adaptive(words_data):
    if not words_data:
        return {}
    avg_height = sum([w['height'] for w in words_data]) / len(words_data)
    tolerance = max(avg_height * 1.2, 15)
    rows = {}
    centroids = {}
    for word in words_data:
        y_center = word['top'] + word['height'] / 2
        found = False
        for row_key in list(rows.keys()):
            if abs(y_center - centroids[row_key]) <= tolerance:
                rows[row_key].append(word)
                centroids[row_key] = sum([(w['top'] + w['height'] / 2) for w in rows[row_key]]) / len(rows[row_key])
                found = True
                break
        if not found:
            rows[y_center] = [word]
            centroids[y_center] = y_center
    return dict(sorted(rows.items()))

def assign_to_nearest_column(center, columns):
    if not columns:
        return 0
    return min(range(len(columns)), key=lambda i: abs(center - columns[i]))

def validate_table_structure(table_rows):
    if not table_rows:
        return False, "No rows detected"
    col_counts = [len(r) for r in table_rows]
    if len(set(col_counts)) > 3:
        return False, f"Inconsistent cols: {set(col_counts)}"
    most = max(set(col_counts), key=col_counts.count)
    if most < 3:
        return False, "Too few columns"
    return True, f"Valid table: {len(table_rows)} rows × {most} cols"

def reconstruct_table_structure(ocr_data, expected_columns=7):
    if not ocr_data or 'text' not in ocr_data:
        return [], False, "No OCR data"
    words = []
    for i in range(len(ocr_data['text'])):
        t = ocr_data['text'][i].strip()
        if t and ocr_data['conf'][i] != -1:
            words.append({
                'text': t,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            })
    if not words:
        return [], False, "No words detected"
    cols = detect_column_centers(words, expected_columns)
    rows = cluster_rows_adaptive(words)
    table = []
    for _, row_words in rows.items():
        row_cells = [''] * len(cols)
        for w in row_words:
            center = w['left'] + w['width'] / 2
            idx = assign_to_nearest_column(center, cols)
            row_cells[idx] += (' ' + w['text']).strip()
        table.append(row_cells)
    is_valid, msg = validate_table_structure(table)
    return table, is_valid, msg

# -----------------------------
# OCR PAGE PROCESSING
# -----------------------------
def extract_text_from_pdf_page(page, dpi=300):
    fitz, err = lazy_import_fitz()
    if err:
        st.error("PyMuPDF import error: " + err)
        return "", None, None

    Image, ImageEnhance, errPIL = lazy_import_PIL()
    if errPIL:
        st.error("PIL import error: " + errPIL)
        return "", None, None

    st.info(f"📄 Rendering page {page.number + 1}")

    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    processed = preprocess_image(image)

    pyt, err = lazy_import_pytesseract()
    if err:
        st.error("Tesseract error: " + err)
        return "", image, None

    try:
        cfg = r'--oem 3 --psm 6'
        text = pyt.image_to_string(processed, lang='asm', config=cfg)
        data = pyt.image_to_data(processed, lang='asm', config=cfg, output_type=pyt.Output.DICT)
        return text, image, data
    except Exception as e:
        st.error(str(e))
        return "", image, None

# -----------------------------
# PDF GENERATION (LAZY REPORTLAB)
# -----------------------------
def create_translated_pdf(original_path, pages, out_path):
    R, err = lazy_import_reportlab()
    if err:
        st.error("ReportLab import failed: " + err)
        return

    doc = R["SimpleDocTemplate"](
        out_path,
        pagesize=R["A4"],
        rightMargin=30, leftMargin=30, topMargin=40, bottomMargin=40
    )

    styles = R["getSampleStyleSheet"]()
    cell_style = R["ParagraphStyle"]("CellText", parent=styles['Normal'], fontSize=7, leading=9)

    story = []
    for i, data in enumerate(pages):
        story.append(R["Paragraph"](f"Page {i+1}", styles['Heading3']))
        if data.get("translated_table"):
            tbl_data = []
            for row in data["translated_table"]:
                tbl_row = [R["Paragraph"](R["escape"](str(c)), cell_style) for c in row]
                tbl_data.append(tbl_row)

            table = R["Table"](tbl_data)
            table.setStyle(R["TableStyle"]([
                ('GRID', (0,0), (-1,-1), 0.5, R["colors"].grey)
            ]))
            story.append(table)
        else:
            story.append(R["Paragraph"]("[No table data]", cell_style))

        if i < len(pages)-1:
            story.append(R["PageBreak"]())

    doc.build(story)

# -----------------------------
# MAIN UI LOGIC (SAFE)
# -----------------------------
def main():
    uploaded_file = st.file_uploader("📤 Upload Assamese PDF", type=['pdf'])

    if not uploaded_file:
        st.info("👆 Please upload an Assamese PDF.")
        return

    # Save temp file
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🚀 Start OCR & Translation", type="primary"):
        fitz, err = lazy_import_fitz()
        if err:
            st.error("PDF engine failed: " + err)
            return

        doc = fitz.open(temp_path)
        total = len(doc)

        progress = st.progress(0)
        status = st.empty()

        pages = []

        # -------------------------
        # STEP 1 — OCR
        # -------------------------
        for i in range(total):
            status.text(f"OCR Page {i+1}/{total}")
            progress.progress((i+1)/(total*3))
            text, image, ocr = extract_text_from_pdf_page(doc[i], dpi=dpi)

            pages.append({
                "page_num": i+1,
                "assamese": text,
                "ocr_data": ocr,
                "table_rows": [],
                "translated_table": []
            })

        # -------------------------
        # STEP 2 — TABLE STRUCTURE
        # -------------------------
        for i, p in enumerate(pages):
            status.text(f"Reconstructing table {i+1}/{total}")
            progress.progress((total+i+1)/(total*3))
            if p["ocr_data"]:
                rows, ok, msg = reconstruct_table_structure(p["ocr_data"])
                p["table_rows"] = rows if ok else []
                p["table_valid"] = ok

        # -------------------------
        # STEP 3 — TRANSLATION
        # -------------------------
        for i, p in enumerate(pages):
            status.text(f"Translating page {i+1}/{total}")
            progress.progress((2*total+i+1)/(total*3))

            if p.get("table_valid"):
                translated = []
                for row in p["table_rows"]:
                    trow = []
                    for cell in row:
                        trow.append(translate_text_with_retry(cell, max_retries=max_retries))
                        time.sleep(0.3)
                    translated.append(trow)
                p["translated_table"] = translated

        progress.progress(1.0)
        status.text("Done!")

        # -------------------------
        # STEP 4 — OUTPUT PDF
        # -------------------------
        out_pdf = f"/tmp/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        create_translated_pdf(temp_path, pages, out_pdf)

        with open(out_pdf, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            "📄 Download Translated PDF",
            pdf_bytes,
            file_name="translated_output.pdf",
            mime="application/pdf"
        )


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
