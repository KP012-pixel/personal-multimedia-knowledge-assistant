# Personal Multimedia Knowledge Assistant

A personal knowledge assistant that helps users search saved video content, ask questions across transcripts, and retrieve source-linked answers using semantic search and local language models.

## Overview

Videos often contain useful information that is difficult to find again. Rewatching entire videos or searching through transcripts takes time, especially when information is spread across multiple sources.

The Personal Multimedia Knowledge Assistant (PMKA) aims to make saved video content searchable and easier to query. Users can organize content into collections, ask natural-language questions, and retrieve answers grounded in relevant saved material.

## Features

- **Video transcription:** Uses Whisper to convert supported video audio into text.
- **Semantic search:** Uses sentence embeddings and FAISS to retrieve content relevant to a question.
- **Question answering:** Uses a locally hosted language model through Ollama to generate answers from retrieved context.
- **Source references:** Returns source information and relevant excerpts to help users trace answers back to saved content.
- **Collections:** Organizes saved content into separate collections.
- **Video and document processing:** Includes backend components for video processing and OCR.
- **Web interface:** Provides a browser-based interface for importing content, viewing collections, and asking questions.

Feature availability depends on the source type, its accessibility, and the current implementation.

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Transcription | OpenAI Whisper |
| Text embeddings | Sentence Transformers |
| Vector search | FAISS |
| Language model integration | Ollama |
| Video processing and OCR | OpenCV, Tesseract |
| YouTube integration | YouTube Data API and Google OAuth |

## Project Structure

```text
personal-multimedia-assistant/
├── backend/
│   ├── main.py
│   ├── answer.py
│   ├── vector_store.py
│   ├── chunker.py
│   ├── index_documents.py
│   ├── ocr.py
│   ├── models.py
│   ├── reranker.py
│   ├── rebuild_index.py
│   ├── requirements.txt
│   └── playlists.example.json
├── frontend/
│   └── index.html
├── .gitignore
└── README.md
```

## How It Works

1. The user imports supported content into a collection.
2. Video audio is transcribed into text where required.
3. Text is processed and divided into searchable segments.
4. Sentence Transformers converts text into numerical embeddings.
5. FAISS searches for segments relevant to the user's question.
6. Relevant context is passed to a local language model through Ollama.
7. The interface presents the generated answer alongside available source references.

The system combines retrieval with language-model generation so that answers can be based on the user's saved content rather than relying only on the model's general knowledge.

## Getting Started

### Prerequisites

Install or configure the following:

- Python and `pip`
- Git
- FFmpeg for Whisper audio processing
- Tesseract OCR if using OCR functionality
- Ollama and a compatible language model
- YouTube API credentials if using YouTube features that require OAuth

### 1. Clone the repository

```bash
git clone https://github.com/KP012-pixel/personal-multimedia-knowledge-assistant.git
cd personal-multimedia-knowledge-assistant
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r backend/requirements.txt
```

Some dependencies, particularly FAISS and Whisper, may require platform-specific setup. Follow their installation instructions if installation fails.

### 4. Configure Ollama

Install Ollama and download the model configured in `backend/answer.py`.

For the current configuration, the model is `llama3:8b`:

```bash
ollama pull llama3:8b
```

Ensure Ollama is running before asking questions.

### 5. Configure external credentials

If using YouTube OAuth, follow the Google Cloud documentation to configure the required API and OAuth credentials. Keep credential files and tokens private. Do not commit them to GitHub.

Refer to the backend configuration and `playlists.example.json` for the project's expected local configuration.

### 6. Start the backend

Open a terminal:

```powershell
cd backend
python -m uvicorn main:app --reload
```

Keep the backend running while using the frontend.

### 7. Open the frontend

Open `frontend/index.html` in a browser. If browser restrictions prevent requests to the backend, serve the frontend through a local HTTP server and ensure the backend's CORS configuration allows that origin.

## How to Use

1. Start the backend and ensure Ollama is running.
2. Open the web interface.
3. Select an existing collection or create one if the interface supports it.
4. Import content using an available source option.
5. Allow processing and indexing to finish.
6. Enter a question about the saved content.
7. Review the answer and its available source references.
8. Open the referenced content to verify the relevant information.

For broad questions, such as asking what topics a collection covers, the assistant may use video titles to provide an overview. Such an overview should be treated as an indication of likely topics, not proof of everything discussed in each video.

## Challenges and Lessons Learned

### 1. Importing content from different platforms

Different platforms expose video content in different ways. YouTube has API and transcript-access options, while Instagram content may require authentication and may not expose transcripts consistently.

This makes a unified import workflow challenging. Source availability, permissions, and transcript access must be handled separately rather than assuming every URL can be processed identically.

### 2. Retrieving relevant information

A user's question may not use the same words as the original video. Semantic embeddings help retrieve relevant passages even when the wording differs, but retrieval quality still depends on transcript quality, segmentation, and the search configuration.

### 3. Keeping answers traceable to sources

A generated answer is more useful when users can inspect its supporting material. Source identifiers, metadata, and relevant excerpts must remain associated with the correct video throughout indexing and retrieval.

### 4. Running language models locally

Local inference reduces reliance on paid hosted model APIs and gives the project more control over its processing workflow. However, model downloads, memory requirements, response time, and local environment configuration introduce their own constraints.

### 5. Managing local data and credentials

Video transcripts, vector indexes, and OAuth tokens have different storage and security requirements. Generated data and private credentials must be kept separate from source code and excluded from public version control where appropriate.

## Current Limitations

- Answer quality depends on the accuracy and completeness of available transcripts.
- Semantic retrieval may miss relevant passages or retrieve incomplete context.
- Video and transcript availability varies by platform.
- Instagram import and other source integrations may be constrained by authentication and access restrictions.
- Locally hosted language models require sufficient system resources.
- Answers and source references should be checked against the original material when accuracy is important.
- The current implementation may not support every planned workflow or content type.

## Future Improvements

- Improve retrieval accuracy and evaluation.
- Strengthen citation validation and timestamp-level source references.
- Improve handling of missing, noisy, or incomplete transcripts.
- Expand reliable import support across platforms.
- Add more robust processing status, error handling, and collection management.
- Evaluate answer quality using a repeatable set of questions and expected source passages.

## Testing

The repository contains Python test files. Run the applicable tests from the backend directory:

```powershell
cd backend
python -m pytest
```

Some tests may require local dependencies, configuration, model downloads, or external services. Test results can vary with the local environment.

## Project Status

PMKA is an evolving project focused on making personal video knowledge searchable through semantic retrieval and locally hosted language models. The implementation and supported import workflows are subject to ongoing development.

## Repository

[Personal Multimedia Knowledge Assistant on GitHub](https://github.com/KP012-pixel/personal-multimedia-knowledge-assistant)
