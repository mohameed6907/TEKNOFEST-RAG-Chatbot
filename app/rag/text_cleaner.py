"""
text_cleaner.py
===============
Text cleaning and normalization for all document types.

Ensures consistent, noise-free text before chunking and embedding.
Handles PDF, DOCX, TXT, and web (HTML-extracted) content.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

DocType = Literal["pdf", "docx", "txt", "web", "md"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex patterns compiled once at import time
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_NULL_BYTES = re.compile(r"\x00+")
_SOFT_HYPHEN = re.compile(r"\xad")
_CONTROL_CHARS = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common PDF header/footer patterns (page numbers, confidential stamps, etc.)
_PDF_PAGE_MARKER = re.compile(
    r"(?i)(page\s+\d+\s+(of\s+\d+)?|^\d+$|\bconfidential\b|\bdraft\b)",
    re.MULTILINE,
)
# Repeated dashes / underscores used as visual separators in PDFs
_VISUAL_SEPARATOR = re.compile(r"[-_=]{4,}")

# Web boilerplate tokens that often appear after HTML stripping
_WEB_BOILERPLATE = re.compile(
    r"(?i)(cookie\s+policy|accept\s+all\s+cookies|copyright\s+©?\s*\d{4}|"
    r"all\s+rights\s+reserved|subscribe\s+to\s+our\s+newsletter|"
    r"follow\s+us\s+on\s+social\s+media|privacy\s+policy|terms\s+of\s+use|"
    r"skip\s+to\s+(main\s+)?content)"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(text: str, doc_type: DocType = "txt") -> str:
    """
    Full cleaning pipeline for a raw document text.

    Steps
    -----
    1. Normalize Unicode to NFC form
    2. Remove null bytes and control characters
    3. Fix soft hyphens (PDF word-wrap artefacts)
    4. Strip doc-type-specific boilerplate
    5. Collapse excessive whitespace / blank lines

    Parameters
    ----------
    text : str
        Raw text as extracted by the document loader.
    doc_type : DocType
        Hint used to apply type-specific cleaning rules.

    Returns
    -------
    str
        Cleaned text ready for chunking.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove null bytes and illegal control chars
    text = _NULL_BYTES.sub("", text)
    text = _CONTROL_CHARS.sub(" ", text)

    # 3. Fix soft hyphens and replacement characters
    text = _SOFT_HYPHEN.sub("", text)
    text = text.replace("\ufffd", "")  # Unicode replacement char

    # 4. Doc-type specific cleaning
    if doc_type == "pdf":
        text = _clean_pdf(text)
    elif doc_type == "web":
        text = _clean_web(text)
    elif doc_type in ("docx", "txt", "md"):
        text = _clean_generic(text)

    # 5. Normalise whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    return text


def normalize_chunk(text: str) -> str:
    """
    Light normalisation applied to each individual chunk just before embedding.

    Strips leading/trailing whitespace, collapses internal runs of spaces, and
    ensures the text is in NFC form.  This is intentionally lightweight —
    heavy cleaning happens in :func:`clean_text` at the document level.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_SPACE.sub(" ", text)
    # Keep paragraph breaks (\n\n) but collapse single newlines to spaces so
    # the embedded representation is more like continuous prose.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def infer_doc_type(filename: str) -> DocType:
    """Infer DocType from a file extension or URL."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx") or name.endswith(".doc"):
        return "docx"
    if name.endswith(".md"):
        return "md"
    if name.startswith("http://") or name.startswith("https://"):
        return "web"
    return "txt"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clean_pdf(text: str) -> str:
    """Remove common PDF artefacts: page numbers, separators, headers."""
    # Remove visual separators
    text = _VISUAL_SEPARATOR.sub("", text)
    # Remove isolated page-number lines
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just a page marker or a lone number
        if _PDF_PAGE_MARKER.fullmatch(stripped):
            continue
        # Skip very short lines that are likely headers/footers (< 4 chars,
        # not empty — empty lines carry paragraph structure)
        if 0 < len(stripped) < 4:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _clean_web(text: str) -> str:
    """Remove web boilerplate injected by BeautifulSoup text extraction."""
    # Remove common navigation / legal boilerplate phrases
    text = _WEB_BOILERPLATE.sub("", text)
    # Remove markdown-style links left by some parsers: [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    return text


def _clean_generic(text: str) -> str:
    """Minimal cleaning for DOCX / TXT / MD files."""
    # Collapse runs of dashes used as section separators
    text = re.sub(r"-{3,}", "—", text)
    return text
