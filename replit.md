# Overview

This is a Streamlit-based PDF OCR and translation application designed specifically for Assamese language documents. The application converts Assamese PDF files to English while preserving the original document layout and structure. It uses Tesseract OCR for text extraction and Google Translate for translation, both completely free with no API keys required. The system includes intelligent rate limiting, automatic retry mechanisms, and real-time progress tracking to ensure reliable processing of multi-page documents.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Framework:** Streamlit web application
- **Rationale:** Provides rapid development of interactive data applications with minimal frontend code
- **Layout:** Wide layout with sidebar for settings and controls
- **Real-time Feedback:** Progress tracking displays current page being processed, word/character counts, and translation status
- **User Controls:** Adjustable settings for processing delays, retry attempts, and OCR quality (DPI)

## Document Processing Pipeline

**Multi-stage Processing:**
1. **PDF Rendering:** Uses PyMuPDF (fitz) to convert PDF pages to high-resolution images
2. **Image Preprocessing:** PIL-based enhancement (grayscale conversion, contrast adjustment) to improve OCR accuracy
3. **OCR Extraction:** Tesseract with Assamese language model for text recognition
4. **Translation:** Google Translate via deep-translator library with retry logic
5. **PDF Generation:** ReportLab for creating translated PDFs with layout preservation

**Problem Addressed:** Assamese text is often misidentified as Bengali by OCR systems
**Solution:** Explicit Tesseract configuration with Assamese language model ('asm' or 'as')
**Alternative Considered:** EasyOCR and PaddleOCR (mentioned in requirements)
**Chosen Approach Pros:** Tesseract is lightweight, well-established, and supports Assamese natively
**Chosen Approach Cons:** May require system-level language data installation

## Rate Limiting & Reliability

**Problem Addressed:** Google Translate free tier has rate limits that cause translation failures
**Solution:** Multi-layered approach:
- Configurable delays between page translations (default 1.5 seconds)
- Exponential backoff retry mechanism with configurable max retries
- Real-time progress tracking to show system is working during delays

**Alternatives Considered:** Paid translation APIs (Google Cloud Translate, Azure)
**Rationale:** User requirement for 100% free solution with zero API keys

## PDF Layout Preservation

**Technology:** ReportLab PDF generation library
- **Problem:** Maintaining original document structure after translation
- **Solution:** Custom TTFont registration for proper text rendering
- **Challenge:** Coordinate mapping from original PDF to translated output
- **Status:** Basic implementation present (TTFont import visible), full layout logic likely in truncated code

## Error Handling

**Comprehensive Error Management:**
- File upload validation (size limits, format checking)
- OCR failure detection (0 character extraction handling)
- Translation retry logic with exponential backoff
- Progress tracking prevents user uncertainty during long operations

# External Dependencies

## OCR Engine
- **Tesseract OCR** - System-level installation required with Assamese language data pack
- **pytesseract** - Python wrapper for Tesseract
- **PyMuPDF (fitz)** - PDF rendering and manipulation

## Image Processing
- **PIL (Pillow)** - Image preprocessing, enhancement, and format conversion

## Translation Service
- **deep-translator** - Free Google Translate wrapper (no API key required)
- Rate-limited, requires intelligent delay management

## PDF Generation
- **ReportLab** - PDF creation with custom font support
- **pdfmetrics/TTFont** - Custom font registration for non-Latin scripts

## Web Framework
- **Streamlit** - Interactive web application framework
- Built-in file upload, progress bars, and sidebar controls

## Python Standard Library
- **io** - In-memory file handling
- **time** - Delay implementation for rate limiting
- **os** - File system operations
- **datetime** - Timestamp tracking for statistics

## System Requirements
- Tesseract OCR installed at system level
- Assamese language data for Tesseract (traineddata file)
- Font files supporting Assamese Unicode characters (for PDF generation)