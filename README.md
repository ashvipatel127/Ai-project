# AI-Powered University Knowledge Assistant

A Python-based Retrieval-Augmented Generation (RAG) system that allows users to ask natural-language questions about university documents and receive relevant answers grounded in the source material.

## Overview

The AI-Powered University Knowledge Assistant processes PDF documents, converts their content into searchable vector representations, and retrieves the most relevant information when a user asks a question.

The project demonstrates a complete document question-answering pipeline using semantic search, embeddings, and vector similarity.

## How It Works

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Embeddings
      ↓
FAISS Vector Search
      ↓
Relevant Document Chunks
      ↓
Answer Generation
      ↓
Answer + Source Citation
```

## Features

* PDF document ingestion
* Page-level text extraction
* Overlapping text chunking
* Semantic text embeddings
* FAISS vector similarity search
* Natural-language question answering
* Source document and page citations
* Demo mode that works without API credits
* OpenAI API integration for future LLM-powered responses

## Technologies Used

* **Python**
* **PyMuPDF** – PDF text extraction
* **Sentence Transformers** – text embeddings
* **FAISS** – vector similarity search
* **OpenAI API** – LLM integration
* **python-dotenv** – environment variable management
* **Git & GitHub** – version control

## Project Structure

```text
Ai-project/
│
├── app/
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   └── pdf_parser.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   └── main.py
│
├── data/
│   └── sample_documents/
│       └── sample_syllabus.pdf
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Example

The system can answer questions about information contained in a course syllabus.

### Question

```text
Who is the instructor?
```

### Answer

```text
Instructor: Dr. Taylor

Source: sample_syllabus.pdf (Page 1)
```

### Question

```text
What percentage of the grade is the final project?
```

### Answer

```text
The final project is worth 45% of the grade.

Source: sample_syllabus.pdf (Page 1)
```

### Question

```text
What is the final project due date?
```

### Answer

```text
The final project is due on December 5.

Source: sample_syllabus.pdf (Page 1)
```

## Installation

Clone the repository:

```bash
git clone (https://github.com/ashvipatel127/Ai-project)
cd Ai-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project includes an `.env.example` file showing the required environment variable.

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your
```

