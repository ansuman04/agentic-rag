
import hashlib
from pathlib import Path

from pypdf import PdfReader


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def load_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Load a PDF and return chunks with source metadata."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    document_id = hashlib.sha256(path.read_bytes()).hexdigest()

    reader = PdfReader(str(path))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        chunks = split_text(
            text.strip(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": (
                        f"{document_id}_page_{page_number}"
                        f"_chunk_{chunk_index}"
                    ),
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "document_id": document_id,
                    },
                }
            )

        print(
            f"Page {page_number}: "
            f"{len(text)} characters, {len(chunks)} chunks"
        )

    if not records:
        raise ValueError("No extractable text found in PDF")

    return records