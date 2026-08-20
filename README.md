# Research Paper Explorer

A lightweight Retrieval-Augmented Generation (RAG) application for asking questions about research papers.

The idea is simple: instead of expecting a language model to know the contents of a paper beforehand, the application takes a PDF supplied by the user, extracts its text, breaks the text into smaller chunks, converts those chunks into vector embeddings, and stores them in a FAISS index. When a question is asked, the system retrieves relevant parts of the paper and uses a local language model to generate an answer.

The project is implemented as a Google Colab notebook and uses open-source models and libraries rather than a paid LLM API.

## What does it do?

You upload a research paper in PDF format and ask a question about it.

For example:

> Upload: `attention_is_all_you_need.pdf`

Then ask:

> What problem does the paper try to solve?

or:

> How does the proposed architecture differ from previous approaches?

The application processes the paper and uses the retrieved sections as context for answering the question.

## How it works

The application follows a basic RAG pipeline:

```text
                Research Paper PDF
                        │
                        ▼
                PDF Text Extraction
                        │
                        ▼
                  Text Chunking
                        │
                        ▼
              Sentence Embeddings
                        │
                        ▼
                  FAISS Index
                        │
                        │
User Question ──────────┘
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
     Local LLM
   (FLAN-T5 Base)
        │
        ▼
      Answer
```

### 1. PDF ingestion

The uploaded PDF is read using `pypdf`. Text is extracted page by page and combined into a single text string.

```python
reader = PdfReader(file_path)

for page in reader.pages:
    text += page.extract_text() + "\n"
```

This gives the rest of the pipeline a plain-text representation of the research paper.

### 2. Text chunking

Large documents are split into smaller pieces before creating embeddings.

The current implementation uses:

* Chunk size: `500`
* Chunk overlap: `100`
* Separator: newline (`\n`)

The overlap helps retain some context between neighboring chunks instead of cutting the document into completely independent pieces.

### 3. Creating embeddings

Each text chunk is converted into a numerical vector using the `all-MiniLM-L6-v2` sentence-transformer model.

These embeddings allow the application to compare the meaning of the user's question with the meaning of different sections of the paper.

In other words, retrieval is based on semantic similarity rather than simply searching for the exact words used in the question.

### 4. FAISS vector index

The generated embeddings are stored in a FAISS vector store.

FAISS is used to efficiently search through the document's vector representations and retrieve chunks that are relevant to a user's question.

The index is created from the document chunks:

```python
faiss_index = FAISS.from_texts(
    chunks,
    embedding=embedder
)
```

### 5. Local language model

The project uses Google's `flan-t5-base` through Hugging Face Transformers.

```python
model_name = "google/flan-t5-base"
```

The model runs locally through a Transformers pipeline rather than sending the document or question to a hosted commercial LLM API.

This makes the project useful for experimenting with RAG without depending on an external paid inference API.

### 6. Retrieval-based question answering

The FAISS index is connected to a LangChain retrieval QA chain.

When a question is submitted, relevant document chunks are retrieved and passed to the language model as context.

The goal is to make the answer depend on the uploaded research paper rather than asking the language model to answer entirely from its pretrained knowledge.

### 7. Gradio interface

The project provides a simple Gradio interface with two inputs:

* Research paper PDF
* User question

The generated answer is displayed directly in the interface.

The notebook uses Gradio's `share=True` option, which can expose the running Colab application through a temporary public link.

## Tech Stack

| Component         | Technology                |
| ----------------- | ------------------------- |
| Language          | Python                    |
| Environment       | Google Colab              |
| PDF processing    | pypdf                     |
| Text processing   | LangChain                 |
| Embeddings        | `all-MiniLM-L6-v2`        |
| Vector database   | FAISS                     |
| Language model    | `google/flan-t5-base`     |
| Model framework   | Hugging Face Transformers |
| RAG orchestration | LangChain                 |
| Interface         | Gradio                    |

## Project Structure

The project is currently implemented as a single Google Colab notebook:

```text
GenAI/
│
└── researchAssistant.ipynb
```

The notebook contains the complete pipeline:

```text
PDF ingestion
    ↓
Text extraction
    ↓
Text splitting
    ↓
Embedding generation
    ↓
FAISS vector index
    ↓
Retriever
    ↓
FLAN-T5
    ↓
Answer
```

## Running the Project

### Option 1: Google Colab

The easiest way to run the project is through Google Colab.

Open the notebook and run the cells from top to bottom.

The notebook installs the required dependencies:

```bash
pip install -q faiss-cpu langchain sentence-transformers transformers
pip install -q pypdf
pip install -U -q langchain-community
pip install -q gradio
```

The original notebook is available here:

[Open Research Assistant in Google Colab](https://colab.research.google.com/github/ARADHYA299/GenAI/blob/main/researchAssistant.ipynb)

### Option 2: Local environment

Clone the repository:

```bash
git clone https://github.com/ARADHYA299/GenAI.git
cd GenAI
```

Install the required packages:

```bash
pip install faiss-cpu langchain sentence-transformers transformers pypdf langchain-community gradio
```

Then open the notebook with Jupyter or run it through an environment that supports the notebook cells.

## Using the Application

Once the Gradio interface is running:

1. Upload a research paper in PDF format.
2. Enter a question about the paper.
3. Submit the question.
4. The application processes the paper and creates its vector index.
5. Relevant chunks are retrieved.
6. FLAN-T5 generates the answer from the retrieved context.

For example:

```text
PDF:
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Question:
What is the main idea behind BERT?
```

Other useful questions include:

```text
What problem does this paper address?

What methodology does the paper propose?

What dataset was used?

What are the main experimental results?

What are the limitations mentioned in the paper?

How does the proposed method compare with previous work?
```

## Why RAG?

A research paper can contain many pages of information, while a language model cannot simply be expected to keep the entire document in its prompt.

RAG separates the problem into two parts:

**Retrieval**

Find the parts of the uploaded document that are relevant to the question.

**Generation**

Give those retrieved sections to a language model and use them to produce an answer.

This makes the system more practical for document-specific question answering.

## Current Limitations

This project is intentionally lightweight, and there are a few things worth knowing before treating it like a production document intelligence platform.

### The PDF is reprocessed for every question

The current Gradio handler processes the uploaded PDF whenever a question is submitted.

That means the document is:

```text
read → split → embedded → indexed → queried
```

again for every question.

This is acceptable for a small prototype but inefficient for longer research papers or repeated questions.

### The vector index is not persisted

The FAISS index currently exists only during the processing session.

A production implementation could save the index and reuse it for subsequent questions instead of rebuilding it.

### Answers depend on PDF extraction quality

`pypdf` extracts text from the PDF directly. Papers containing complicated layouts, scanned pages, tables, mathematical notation, or unusual formatting may not extract cleanly.

That can affect retrieval and therefore the final answer.

### The model is relatively small

The project uses `google/flan-t5-base`.

This keeps the project relatively lightweight, but it also means the quality of generated answers can be limited compared with larger instruction-tuned models.

### Source handling can be improved

The retrieval chain is configured to return source documents, but the current interface only displays the generated answer.

A future version could show the retrieved passages alongside the answer so that users can see where the response came from.

## Possible Improvements

There are several straightforward directions for extending the project.

### 1. Persist FAISS indexes

Instead of creating a new index for every question, generate the index once when the PDF is uploaded and reuse it.

### 2. Improve document chunking

Experiment with:

* Recursive character splitting
* Larger or smaller chunks
* Different overlap sizes
* Section-aware chunking

Research papers have structure, so blindly splitting every 500 characters is not always ideal.

### 3. Add source citations

Return the retrieved document chunks with the answer.

The interface could display:

```text
Answer
──────
...

Sources
───────
Page 4
Page 7
Page 12
```

This would make the system much more useful for actual research work.

### 4. Support multiple documents

Instead of restricting the system to one paper, the vector store could contain several papers.

Users could then ask questions such as:

```text
How do these two papers approach the same problem differently?
```

### 5. Add conversation history

A conversational interface could allow follow-up questions such as:

```text
User:
What is the proposed architecture?

User:
Why did the authors choose it?

User:
What were the main experimental results?
```

without requiring the user to restate the context each time.

### 6. Improve the user interface

The current Gradio interface is intentionally minimal. A more complete version could include:

* Document information
* Retrieved passages
* Source pages
* Conversation history
* Loading/progress indicators
* Multiple document support

## Project Goal

The project was built to understand and demonstrate the core components of a Retrieval-Augmented Generation system rather than simply calling an LLM API and displaying its response.

The important part of the project is the pipeline:

```text
Unstructured document
        ↓
Text extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector search
        ↓
Relevant context
        ↓
Language model
        ↓
Document-grounded answer
```

That pipeline is the foundation for many larger document-question-answering systems.

## Repository

GitHub:

https://github.com/ARADHYA299/GenAI

Notebook:

https://github.com/ARADHYA299/GenAI/blob/main/researchAssistant.ipynb

## Author

**Aradhya Nautiyal**

GitHub: https://github.com/ARADHYA299

LinkedIn: https://www.linkedin.com/in/aradhya-nautiyal-54bab0268/
