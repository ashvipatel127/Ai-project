from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of dictionaries containing:
    - document name
    - page number
    - extracted text
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    document = fitz.open(pdf_file)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "document": pdf_file.name,
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages


if __name__ == "__main__":
    print("PDF parser loaded successfully.")