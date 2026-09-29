
import os
import re
from openai import OpenAI


class LLMGenerator:
    """
    Generates answers using either:
    1. OpenAI API
    2. Free demo/test mode

    Demo mode does not require API credits.
    """

    def __init__(self):
        self.demo_mode = True

        if not self.demo_mode:
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )

            self.client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        question: str,
        context_documents: list[dict],
    ) -> str:

        if not context_documents:
            return (
                "I couldn't find enough information in the "
                "provided documents to answer that question."
            )

        # --------------------------------------------------
        # FREE DEMO MODE
        # --------------------------------------------------

        if self.demo_mode:

            question_lower = question.lower()

            all_text = "\n".join(
                document["text"]
                for document in context_documents
            )

            # --------------------------------------------------
            # Specific question patterns
            # --------------------------------------------------

            # Instructor
            if "instructor" in question_lower:
                match = re.search(
                    r"Instructor:\s*(.+)",
                    all_text,
                    re.IGNORECASE
                )

                if match:
                    answer = match.group(1).strip()

                    return (
                        f"Instructor: {answer}\n\n"
                        f"Source: {context_documents[0]['document']} "
                        f"(Page {context_documents[0]['page']})"
                    )

            # Final project due date
            if (
                "due date" in question_lower
                or "when is the final project due" in question_lower
            ):
                match = re.search(
                    r"Final Project:\s*(?:\n)?"
                    r"The final project is due on\s+(.+?)(?:\.|\n|$)",
                    all_text,
                    re.IGNORECASE
                )

                if match:
                    answer = match.group(1).strip()

                    return (
                        f"The final project is due on {answer}.\n\n"
                        f"Source: {context_documents[0]['document']} "
                        f"(Page {context_documents[0]['page']})"
                    )

            # Final project percentage
            if (
                "percentage" in question_lower
                or "percent" in question_lower
                or "grade" in question_lower
            ):
                match = re.search(
                    r"Final Project:\s*(\d+%)",
                    all_text,
                    re.IGNORECASE
                )

                if match:
                    answer = match.group(1)

                    return (
                        f"The final project is worth {answer} of the grade.\n\n"
                        f"Source: {context_documents[0]['document']} "
                        f"(Page {context_documents[0]['page']})"
                    )

            # --------------------------------------------------
            # General fallback
            # --------------------------------------------------

            sentences = re.split(
                r"(?<=[.!?])\s+|\n+",
                all_text
            )

            question_words = set(
                re.findall(
                    r"\b[a-zA-Z]{3,}\b",
                    question_lower
                )
            )

            best_sentence = None
            best_score = 0

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                sentence_words = set(
                    re.findall(
                        r"\b[a-zA-Z]{3,}\b",
                        sentence.lower()
                    )
                )

                overlap = question_words.intersection(
                    sentence_words
                )

                score = len(overlap)

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            if best_sentence and best_score > 0:

                return (
                    f"{best_sentence}\n\n"
                    f"Source: {context_documents[0]['document']} "
                    f"(Page {context_documents[0]['page']})"
                )

            return (
                "I couldn't find enough information in the "
                "provided documents to answer that question."
            )

        # --------------------------------------------------
        # OPENAI MODE
        # --------------------------------------------------

        context_parts = []

        for document in context_documents:
            context_parts.append(
                f"""
Source: {document['document']}
Page: {document['page']}

{document['text']}
"""
            )

        context = "\n---\n".join(context_parts)

        prompt = f"""
You are a university knowledge assistant.

Answer the user's question using ONLY the information
contained in the provided source material.

If the source material does not contain enough information
to answer the question, say:

"I couldn't find enough information in the provided documents
to answer that question."

Do not invent facts.

User question:
{question}

Source material:
{context}
"""

        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
        )

        return response.output_text

