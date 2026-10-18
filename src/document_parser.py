"""
document_parser.py
------------------
Document parsing module for the AI Content & Plagiarism Detection System.

Supports:
  - Plain text (.txt)
  - PDF files (.pdf) via pypdf
  - Microsoft Word documents (.docx) via python-docx
"""

from __future__ import annotations

import io


def parse_txt(file_bytes: bytes, encoding: str = "utf-8") -> str:
    """
    Decode plain text file bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes read from the .txt file.
    encoding : str
        Character encoding to attempt first (default: utf-8).

    Returns
    -------
    str
        Decoded text content.
    """
    for enc in (encoding, "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file using pypdf.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages.

    Raises
    ------
    RuntimeError
        If pypdf is not installed or PDF cannot be parsed.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing requires 'pypdf'. Install it with: pip install pypdf"
            ) from exc

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    if not pages:
        raise ValueError("No text could be extracted from the PDF. The file may be scanned or image-based.")

    return "\n".join(pages)


def parse_docx(file_bytes: bytes) -> str:
    """
    Extract text from a .docx file using python-docx.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the .docx file.

    Returns
    -------
    str
        Concatenated paragraph text.

    Raises
    ------
    RuntimeError
        If python-docx is not installed.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DOCX parsing requires 'python-docx'. Install it with: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("No text could be extracted from the DOCX file.")
    return "\n".join(paragraphs)


def parse_document(filename: str, file_bytes: bytes) -> str:
    """
    Dispatch to the appropriate parser based on the file extension.

    Parameters
    ----------
    filename : str
        Original filename (used to detect extension).
    file_bytes : bytes
        Raw file content.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    parsers = {
        "txt": parse_txt,
        "pdf": parse_pdf,
        "docx": parse_docx,
    }

    if ext not in parsers:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported formats: {', '.join(parsers.keys())}."
        )

    return parsers[ext](file_bytes)


def load_documents(uploaded_files: list) -> list[dict]:
    """
    Parse a list of Streamlit UploadedFile objects.

    Parameters
    ----------
    uploaded_files : list
        List of st.UploadedFile objects from st.file_uploader.

    Returns
    -------
    list[dict]
        Each element is a dict with keys:
          - 'name'  : filename string
          - 'text'  : extracted raw text
          - 'error' : None if successful, error message string otherwise
    """
    results: list[dict] = []
    for uf in uploaded_files:
        entry: dict = {"name": uf.name, "text": "", "error": None}
        try:
            raw_bytes = uf.read()
            if not raw_bytes:
                raise ValueError("The file appears to be empty.")
            text = parse_document(uf.name, raw_bytes)
            if not text.strip():
                raise ValueError("No readable text found in this file.")
            entry["text"] = text
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        results.append(entry)
    return results

