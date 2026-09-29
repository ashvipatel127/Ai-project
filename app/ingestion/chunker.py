def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split page text into smaller overlapping chunks.

    Each chunk keeps the original document and page metadata.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "document": page["document"],
                        "page": page["page"],
                        "text": chunk,
                    }
                )

            start += chunk_size - chunk_overlap

    return chunks