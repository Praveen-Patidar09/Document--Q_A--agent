"""Text extraction with structural metadata. Image-only PDFs are rejected explicitly."""

import csv
import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath

from app.models import DocumentPart

SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm"}
MAX_UNPACKED_BYTES = 100 * 1024 * 1024


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Keep paragraph boundaries; collapse only runs of blank lines.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _check_archive(content: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_UNPACKED_BYTES:
            raise ValueError("This file is too large once unpacked.")


def _table_rows(table) -> list[str]:
    return [" | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
            for row in table.rows]


def _slide_text(shapes) -> list[str]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    result = []
    for shape in shapes:
        try:
            grouped = shape.shape_type == MSO_SHAPE_TYPE.GROUP
        except NotImplementedError:
            grouped = False
        if grouped:
            result.extend(_slide_text(shape.shapes))
        elif getattr(shape, "has_table", False) and shape.has_table:
            result.extend(_table_rows(shape.table))
        elif getattr(shape, "has_text_frame", False) and shape.text.strip():
            result.append(shape.text.strip())
    return result


def _part(text: str, **metadata) -> list[DocumentPart]:
    text = normalize_text(text)
    return [DocumentPart(text, metadata)] if text else []


def extract_document(filename: str, content: bytes) -> list[DocumentPart]:
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED))}")
    if not content:
        raise ValueError("The uploaded file is empty.")

    try:
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError("This file does not appear to be a valid PDF.")
            from pypdf import PdfReader

            pdf = PdfReader(io.BytesIO(content), strict=False)
            if pdf.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")
            parts = []
            for page_no, page in enumerate(pdf.pages, 1):
                parts.extend(_part(page.extract_text() or "", page=page_no))
            return parts

        if suffix in {".txt", ".md"}:
            return _part(content.decode("utf-8-sig", errors="replace"))

        if suffix == ".csv":
            rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
            if not rows:
                return []
            headers = [value.strip() or f"Column {i + 1}" for i, value in enumerate(rows[0])]
            lines = [" | ".join(headers)]
            for row in rows[1:]:
                values = [f"{headers[i]}: {value}" for i, value in enumerate(row[:len(headers)])]
                if values:
                    lines.append(" | ".join(values))
            return _part("\n".join(lines), structure="CSV rows", columns=" | ".join(headers))

        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")
            for element in soup(["script", "style", "noscript", "svg"]):
                element.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            headings = [node.get_text(" ", strip=True) for node in soup.find_all(re.compile(r"^h[1-6]$"))]
            blocks = []
            for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
                value = node.get_text(" ", strip=True)
                if value:
                    blocks.append(value)
            text = "\n\n".join(blocks) or soup.get_text("\n", strip=True)
            return _part(text, title=title[:500], headings=headings[:100])

        if suffix == ".docx":
            if not content.startswith(b"PK"):
                raise ValueError("This file does not appear to be a valid DOCX document.")
            _check_archive(content)
            from docx import Document as WordDocument
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            word = WordDocument(io.BytesIO(content))
            parts = []
            heading = None
            buffer = []

            def flush():
                if buffer:
                    parts.extend(_part("\n\n".join(buffer), heading=heading))
                    buffer.clear()

            for child in word.element.body.iterchildren():
                if child.tag.endswith("}p"):
                    paragraph = Paragraph(child, word)
                    value = paragraph.text.strip()
                    if not value:
                        continue
                    if paragraph.style and paragraph.style.name.startswith("Heading"):
                        flush()
                        heading = value
                    else:
                        buffer.append(value)
                elif child.tag.endswith("}tbl"):
                    table = Table(child, word)
                    buffer.extend(_table_rows(table))
            flush()
            return parts

        if suffix == ".pptx":
            if not content.startswith(b"PK"):
                raise ValueError("This file does not appear to be a valid PPTX presentation.")
            _check_archive(content)
            from pptx import Presentation

            presentation = Presentation(io.BytesIO(content))
            result = []
            for number, slide in enumerate(presentation.slides, 1):
                result.extend(_part("\n\n".join(_slide_text(slide.shapes)), slide=number))
            return result

        if suffix == ".xlsx":
            if not content.startswith(b"PK"):
                raise ValueError("This file does not appear to be a valid XLSX workbook.")
            _check_archive(content)
            from openpyxl import load_workbook

            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            result = []
            try:
                for sheet in workbook.worksheets:
                    rows = [["" if value is None else str(value).strip() for value in row]
                            for row in sheet.iter_rows(values_only=True)]
                    rows = [row for row in rows if any(row)]
                    if not rows:
                        continue
                    headers = [cell or f"Column {i + 1}" for i, cell in enumerate(rows[0])]
                    lines = [" | ".join(headers)]
                    for row in rows[1:]:
                        lines.append(" | ".join(f"{headers[i]}: {value}"
                                                 for i, value in enumerate(row[:len(headers)])))
                    result.extend(_part("\n".join(lines), sheet=sheet.title,
                                        columns=" | ".join(headers)))
            finally:
                workbook.close()
            return result
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The file is corrupt or could not be parsed.") from exc
    return []
