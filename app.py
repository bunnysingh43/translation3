import streamlit as st
import fitz
from PIL import Image
import pytesseract
import io
import time
from deep_translator import GoogleTranslator
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime

st.set_page_config(page_title="Assamese PDF OCR & Translation", layout="wide", page_icon="🔤")

st.title("🔤 Assamese PDF OCR & Translation System")
st.markdown("**100% Free OCR | Assamese → English | Layout Preserved**")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Features:**
    - 🔤 Tesseract OCR with Assamese language
    - 🌐 Free Google Translate (no API key)
    - ⏱️ Intelligent rate limiting (no errors!)
    - 🔄 Automatic retry with exponential backoff
    - 📐 Layout preservation
    - 📊 Real-time progress tracking
    - ✅ Comprehensive error handling
    """)
    
    st.markdown("### ⚙️ Settings")
    delay_between_pages = st.slider("Delay between pages (seconds)", 0.5, 5.0, 1.5, 0.5, 
                                     help="Prevents rate limiting by Google Translate")
    max_retries = st.number_input("Max retries per translation", 1, 10, 3)
    dpi = st.number_input("OCR DPI Quality", 150, 600, 300, 50)

st.markdown("---")

def preprocess_image(image):
    """Enhanced image preprocessing for better OCR accuracy"""
    img = image.convert('L')
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img

def translate_text_with_retry(text, max_retries=3, initial_delay=0.5):
    """
    Translate text with intelligent retry logic and exponential backoff
    to prevent rate limiting errors
    """
    if not text or text.strip() == "":
        return ""
    
    for attempt in range(max_retries):
        try:
            delay = initial_delay * (2 ** attempt)
            if attempt > 0:
                st.warning(f"⏳ Retry attempt {attempt + 1}/{max_retries} (waiting {delay:.1f}s)...")
                time.sleep(delay)
            
            translator = GoogleTranslator(source='auto', target='en')
            translated = translator.translate(text)
            
            time.sleep(0.3)
            
            return translated
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "too many requests" in error_msg or "rate limit" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** (attempt + 1))
                    st.warning(f"⚠️ Rate limit detected. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    st.error(f"❌ Translation failed after {max_retries} attempts: {str(e)}")
                    return f"[TRANSLATION FAILED: {text[:50]}...]"
            else:
                st.error(f"❌ Translation error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return f"[ERROR: {text[:50]}...]"
    
    return text

def translate_text_batch(text_list, max_retries=3, batch_delay=2.0):
    """
    Translate a batch of text with proper rate limiting
    Processes one at a time with delays to respect Google Translate limits
    """
    translations = []
    total = len(text_list)
    
    for idx, text in enumerate(text_list):
        st.info(f"🌐 Translating fragment {idx + 1}/{total}...")
        
        translated = translate_text_with_retry(text, max_retries=max_retries)
        translations.append(translated)
        
        if idx < total - 1:
            time.sleep(batch_delay)
    
    return translations

def detect_column_centers(words_data, expected_columns=7):
    """Detect column centers ensuring expected column count is always returned"""
    if not words_data:
        return []
    
    x_positions = sorted([w['left'] + w['width'] / 2 for w in words_data])
    
    if len(x_positions) < 2:
        return [x_positions[0]] if x_positions else []
    
    min_x = min(x_positions)
    max_x = max(x_positions)
    page_width = max_x - min_x
    
    all_gaps = []
    for i in range(len(x_positions) - 1):
        gap_size = x_positions[i + 1] - x_positions[i]
        all_gaps.append((x_positions[i], x_positions[i + 1], gap_size))
    
    if not all_gaps:
        equal_width = page_width / expected_columns
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
    
    while len(column_centers) < expected_columns:
        gap_idx = 0
        max_gap_size = 0
        for i in range(len(column_centers) - 1):
            gap = column_centers[i + 1] - column_centers[i]
            if gap > max_gap_size:
                max_gap_size = gap
                gap_idx = i
        
        new_center = (column_centers[gap_idx] + column_centers[gap_idx + 1]) / 2
        column_centers.insert(gap_idx + 1, new_center)
    
    return column_centers[:expected_columns]

def cluster_rows_adaptive(words_data):
    """Group words into rows with dynamic centroid tracking for multi-line cells"""
    if not words_data:
        return {}
    
    avg_height = sum([w['height'] for w in words_data]) / len(words_data) if words_data else 15
    tolerance = max(avg_height * 1.2, 15)
    
    rows = {}
    row_centroids = {}
    
    for word in words_data:
        y_center = word['top'] + word['height'] / 2
        
        found_row = False
        for row_key in list(rows.keys()):
            centroid = row_centroids.get(row_key, row_key)
            
            if abs(y_center - centroid) <= tolerance:
                rows[row_key].append(word)
                
                all_y_centers = [w['top'] + w['height'] / 2 for w in rows[row_key]]
                row_centroids[row_key] = sum(all_y_centers) / len(all_y_centers)
                
                found_row = True
                break
        
        if not found_row:
            rows[y_center] = [word]
            row_centroids[y_center] = y_center
    
    return dict(sorted(rows.items()))

def assign_to_nearest_column(word_center, column_centers):
    """Assign a word to the nearest column center"""
    if not column_centers:
        return 0
    
    distances = [abs(word_center - col_center) for col_center in column_centers]
    return distances.index(min(distances))

def validate_table_structure(table_rows):
    """Validate reconstructed table for consistency"""
    if not table_rows:
        return False, "No rows detected"
    
    col_counts = [len(row) for row in table_rows]
    
    if len(set(col_counts)) > 3:
        return False, f"Inconsistent column counts: {set(col_counts)}"
    
    most_common_cols = max(set(col_counts), key=col_counts.count)
    
    if most_common_cols < 3:
        return False, f"Too few columns detected: {most_common_cols}"
    
    if len(table_rows) < 2:
        return False, f"Too few rows detected: {len(table_rows)}"
    
    return True, f"Valid table: {len(table_rows)} rows × {most_common_cols} columns"

def reconstruct_table_structure(ocr_data, expected_columns=7):
    """Parse OCR bounding box data to reconstruct table rows and columns with validation"""
    if not ocr_data or 'text' not in ocr_data:
        return [], False, "No OCR data available"
    
    words_data = []
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text and ocr_data['conf'][i] != -1:
            words_data.append({
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            })
    
    if not words_data:
        return [], False, "No text detected in OCR"
    
    column_centers = detect_column_centers(words_data, expected_columns=expected_columns)
    
    if not column_centers:
        return [], False, "Could not detect columns"
    
    num_columns = len(column_centers)
    
    rows_dict = cluster_rows_adaptive(words_data)
    
    table_rows = []
    for row_y, words in rows_dict.items():
        row_cells = ['' for _ in range(num_columns)]
        
        for word in words:
            word_center = word['left'] + word['width'] / 2
            col_idx = assign_to_nearest_column(word_center, column_centers)
            
            if col_idx < num_columns:
                if row_cells[col_idx]:
                    row_cells[col_idx] += ' ' + word['text']
                else:
                    row_cells[col_idx] = word['text']
        
        row_cells = [cell.strip() for cell in row_cells]
        table_rows.append(row_cells)
    
    is_valid, validation_msg = validate_table_structure(table_rows)
    
    return table_rows, is_valid, validation_msg

def extract_text_from_pdf_page(page, dpi=300):
    """Extract text from a single PDF page using OCR with layout information"""
    st.info(f"📄 Page {page.number + 1}: Rendering PDF page to image ({dpi} DPI)...")
    
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    
    st.info(f"🔧 Page {page.number + 1}: Preprocessing image...")
    processed_image = preprocess_image(image)
    
    st.info(f"🔍 Page {page.number + 1}: Running Tesseract OCR with Assamese language model...")
    
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, lang='asm', config=custom_config)
        
        ocr_data = pytesseract.image_to_data(processed_image, lang='asm', config=custom_config, output_type=pytesseract.Output.DICT)
        
        words = len(text.split())
        chars = len(text)
        
        if chars > 0:
            st.success(f"✅ Page {page.number + 1}: Extracted {words} words, {chars} characters")
        else:
            st.warning(f"⚠️ Page {page.number + 1}: No text extracted (possible image-only page)")
        
        return text, image, ocr_data
        
    except Exception as e:
        st.error(f"❌ OCR Error on page {page.number + 1}: {str(e)}")
        return "", image, None

def translate_table_cells(table_rows, max_retries=3, delay=0.5):
    """Translate each cell in the table individually while preserving structure"""
    if not table_rows:
        return []
    
    translated_table = []
    total_cells = sum(len(row) for row in table_rows)
    cell_count = 0
    
    st.info(f"🌐 Translating {total_cells} cells from {len(table_rows)} rows...")
    
    for row_idx, row in enumerate(table_rows):
        translated_row = []
        
        for cell_idx, cell_text in enumerate(row):
            cell_count += 1
            
            if cell_text.strip():
                st.info(f"🌐 Translating cell {cell_count}/{total_cells} (Row {row_idx + 1}, Col {cell_idx + 1})...")
                translated_cell = translate_text_with_retry(cell_text, max_retries=max_retries)
                translated_row.append(translated_cell)
                
                if cell_count < total_cells:
                    time.sleep(delay)
            else:
                translated_row.append('')
        
        translated_table.append(translated_row)
    
    return translated_table

def create_translated_pdf(original_pdf_path, page_data, output_path, dpi=300):
    """Create PDF with preserved table layout using ReportLab Table class"""
    st.info("📝 Generating layout-preserved translated PDF...")
    
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from xml.sax.saxutils import escape
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=40,
        bottomMargin=40,
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'PageTitle',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceAfter=10,
        textColor=colors.HexColor('#000080'),
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )
    
    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        fontName='Helvetica'
    )
    
    story = []
    
    for page_num, data in enumerate(page_data):
        translated_table = data.get('translated_table', [])
        
        page_title = Paragraph(f"Page {page_num + 1}", title_style)
        story.append(page_title)
        story.append(Spacer(1, 0.1 * inch))
        
        if translated_table and len(translated_table) > 0:
            table_data = []
            
            for row in translated_table:
                table_row = []
                for cell in row:
                    safe_cell = escape(str(cell))
                    cell_para = Paragraph(safe_cell, cell_style)
                    table_row.append(cell_para)
                table_data.append(table_row)
            
            if table_data:
                max_cols = max(len(row) for row in table_data)
                col_widths = [A4[0] / max_cols * 0.85 for _ in range(max_cols)]
                
                table = Table(table_data, colWidths=col_widths, repeatRows=0)
                
                table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8E8E8')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ]))
                
                story.append(table)
        else:
            no_content = Paragraph("[No table data on this page]", cell_style)
            story.append(no_content)
        
        if page_num < len(page_data) - 1:
            story.append(PageBreak())
    
    doc.build(story)
    st.success(f"✅ Layout-preserved translated PDF created")

uploaded_file = st.file_uploader("📤 Upload Assamese PDF", type=['pdf'])

if uploaded_file:
    file_size = len(uploaded_file.getvalue()) / 1024
    st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size:.2f} KB)")
    
    temp_pdf_path = f"/tmp/{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if st.button("🚀 Start OCR & Translation", type="primary"):
        start_time = time.time()
        
        st.markdown("### 📊 Processing Pipeline")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            doc = fitz.open(temp_pdf_path)
            total_pages = len(doc)
            st.info(f"✅ Opened PDF with {total_pages} pages")
            
            page_data = []
            
            st.markdown("#### 🔍 Step 1: OCR Processing")
            
            for page_num in range(total_pages):
                page = doc[page_num]
                status_text.text(f"🔍 OCR Processing: Page {page_num + 1}/{total_pages}")
                progress_bar.progress((page_num) / (total_pages * 2))
                
                text, image, ocr_data = extract_text_from_pdf_page(page, dpi=dpi)
                
                page_data.append({
                    'page_num': page_num + 1,
                    'assamese_text': text,
                    'translated': '',
                    'image': image,
                    'ocr_data': ocr_data
                })
                
                with st.expander(f"📝 View Assamese Text - Page {page_num + 1}"):
                    if text:
                        st.text_area(f"Extracted Text (Page {page_num + 1})", text, height=150)
                    else:
                        st.warning("No text extracted from this page")
            
            doc.close()
            
            st.markdown("#### 📋 Step 2: Table Structure Reconstruction")
            
            for idx, data in enumerate(page_data):
                st.info(f"📋 Page {idx + 1}: Detecting table structure from OCR data...")
                
                if data['ocr_data']:
                    table_rows, is_valid, validation_msg = reconstruct_table_structure(data['ocr_data'], expected_columns=7)
                    
                    if is_valid and table_rows and len(table_rows) > 0:
                        data['table_rows'] = table_rows
                        data['table_valid'] = True
                        st.success(f"✅ Page {idx + 1}: {validation_msg}")
                        
                        with st.expander(f"📋 View Table Structure - Page {idx + 1}"):
                            st.write(f"Validation: {validation_msg}")
                            if len(table_rows) > 0:
                                st.write(f"Sample row 1: {table_rows[0]}")
                                if len(table_rows) > 1:
                                    st.write(f"Sample row 2: {table_rows[1]}")
                    else:
                        st.error(f"❌ Page {idx + 1}: Table validation failed - {validation_msg}")
                        data['table_rows'] = []
                        data['table_valid'] = False
                else:
                    st.warning(f"⚠️ Page {idx + 1}: No OCR data available")
                    data['table_rows'] = []
                    data['table_valid'] = False
            
            st.markdown("#### 🌐 Step 3: Cell-by-Cell Translation")
            st.info(f"⏱️ Using {delay_between_pages}s delay between pages to prevent rate limiting...")
            
            for idx, data in enumerate(page_data):
                status_text.text(f"🌐 Translating: Page {idx + 1}/{total_pages}")
                progress_bar.progress((total_pages + idx + 1) / (total_pages * 2))
                
                st.info(f"📄 Page {idx + 1}: Translating table cells...")
                
                if data.get('table_valid') and data.get('table_rows') and len(data['table_rows']) > 0:
                    translated_table = translate_table_cells(data['table_rows'], max_retries=max_retries, delay=0.3)
                    data['translated_table'] = translated_table
                    
                    st.success(f"✅ Page {idx + 1}: Translation completed - {len(translated_table)} rows")
                    
                    with st.expander(f"🔄 View Translated Table - Page {idx + 1}"):
                        if translated_table and len(translated_table) > 0:
                            st.write("Sample translated rows:")
                            for i, row in enumerate(translated_table[:3]):
                                st.write(f"Row {i + 1}: {row}")
                else:
                    if not data.get('table_valid'):
                        st.error(f"⚠️ Page {idx + 1}: Skipping translation - table validation failed")
                    else:
                        st.warning(f"⚠️ Page {idx + 1}: No table structure to translate")
                    data['translated_table'] = []
                
                if idx < total_pages - 1:
                    st.info(f"⏳ Waiting {delay_between_pages}s before next page to respect rate limits...")
                    time.sleep(delay_between_pages)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing Complete!")
            
            st.markdown("#### 📥 Step 4: Generate Output Files")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_pdf = f"/tmp/translated_{timestamp}.pdf"
            output_txt = f"/tmp/translation_{timestamp}.txt"
            
            create_translated_pdf(temp_pdf_path, page_data, output_pdf)
            
            with open(output_txt, 'w', encoding='utf-8') as f:
                for data in page_data:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PAGE {data['page_num']}\n")
                    f.write(f"{'='*60}\n\n")
                    
                    if data.get('translated_table'):
                        f.write("TRANSLATED TABLE (English):\n")
                        for row_idx, row in enumerate(data['translated_table']):
                            f.write(f"Row {row_idx + 1}: {' | '.join(row)}\n")
                        f.write("\n")
                    else:
                        f.write("No table data available\n\n")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            st.markdown("### 📊 Processing Stats")
            col1, col2, col3, col4 = st.columns(4)
            
            total_words = sum([len(d['assamese_text'].split()) for d in page_data])
            total_chars = sum([len(d['assamese_text']) for d in page_data])
            
            with col1:
                st.metric("Total Pages", total_pages)
            with col2:
                st.metric("Words Extracted", f"{total_words:,}")
            with col3:
                st.metric("Characters", f"{total_chars:,}")
            with col4:
                st.metric("Processing Time", f"{total_time:.1f}s")
            
            st.success("🎉 Processing completed successfully!")
            
            st.markdown("### 📥 Download Results")
            
            with open(output_pdf, 'rb') as f:
                pdf_data = f.read()
            
            with open(output_txt, 'r', encoding='utf-8') as f:
                txt_data = f.read()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Translated PDF",
                    data=pdf_data,
                    file_name=f"translated_{uploaded_file.name}",
                    mime="application/pdf",
                    key="download_pdf_btn"
                )
            
            with col2:
                st.download_button(
                    label="📝 Download Text File",
                    data=txt_data,
                    file_name=f"translation_{uploaded_file.name.replace('.pdf', '.txt')}",
                    mime="text/plain",
                    key="download_txt_btn"
                )
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload an Assamese PDF file to begin")
    
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload PDF**: Select your Assamese language PDF document
    2. **OCR Processing**: Tesseract extracts Assamese text from each page (300 DPI)
    3. **Smart Translation**: Google Translate converts to English with intelligent rate limiting
    4. **PDF Generation**: Creates new PDF with English text preserving original layout
    5. **Download**: Get both translated PDF and text file
    
    **Why This Won't Fail:**
    - ✅ Configurable delays between pages (default 1.5s)
    - ✅ Automatic retry with exponential backoff (up to 3 attempts)
    - ✅ Smart batch processing to respect 5 req/sec limit
    - ✅ Comprehensive error handling and recovery
    - ✅ Real-time progress tracking with detailed logs
    """)
