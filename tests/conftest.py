
import pytest
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf(tmp_path):
    """Generate a small PDF for automated tests."""

    pdf_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(
        50,
        750,
        "Transformers use positional encoding "
        "to represent token order.",
    )

    pdf.showPage()

    pdf.drawString(
        50,
        750,
        "Self-attention allows tokens to attend "
        "to other tokens in a sequence.",
    )

    pdf.showPage()
    pdf.save()

    return pdf_path