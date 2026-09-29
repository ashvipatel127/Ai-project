print("MAIN.PY IS RUNNING")
from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.generation.llm import LLMGenerator


load_dotenv()


def build_knowledge_base(pdf_directory: str):
    """
    Load PDFs, extract text, create chunks,
    generate embeddings, and build the vector store.
    """

    pdf_directory = Path(pdf_directory)

    all_chunks = []

    for pdf_file in pdf_directory.glob("*.pdf"):

        print(f"Processing: {pdf_file.name}")

        pages = extract_text_from_pdf(str(pdf_file))

        chunks = chunk_text(pages)

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No PDF documents were found."
        )

    print(f"Created {len(all_chunks)} text chunks.")

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(texts)

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        all_chunks,
    )

    return embedding_model, vector_store


def ask_question(
    question: str,
    embedding_model,
    vector_store,
    llm_generator,
):
    """
    Retrieve relevant information and generate
    a grounded answer.
    """

    query_embedding = embedding_model.encode(
        [question]
    )

    results = vector_store.search(
        query_embedding[0],
        top_k=5,
    )

    print("\nRetrieved sources:")

    for result in results:

        print(
            f"- {result['document']} "
            f"(Page {result['page']}, "
            f"Score: {result['score']:.3f})"
        )

    answer = llm_generator.generate_answer(
        question,
        results,
    )

    return answer, results


def main():

    print("=" * 60)
    print("AI-POWERED UNIVERSITY KNOWLEDGE ASSISTANT")
    print("=" * 60)

    embedding_model, vector_store = build_knowledge_base(
        "data/sample_documents"
    )

    llm_generator = LLMGenerator()

    print("\nKnowledge base ready!")

    while True:

        question = input(
            "\nAsk a question (or type 'quit'): "
        ).strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer, sources = ask_question(
            question,
            embedding_model,
            vector_store,
            llm_generator,
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(answer)

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        seen_sources = set()

        for source in sources:

            source_key = (
                source["document"],
                source["page"],
            )

            if source_key not in seen_sources:

                print(
                    f"📄 {source['document']} "
                    f"— Page {source['page']}"
                )

                seen_sources.add(source_key)


if __name__ == "__main__":
    main()