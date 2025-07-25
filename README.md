﻿# GenAI Research Assistant

A modern, offline-capable research assistant app built with Streamlit. Upload a PDF or TXT document to get instant summaries, ask questions, or take a logic-based quiz generated from your document—all without sending your data to the cloud.

## Features

- **Document Upload:** Supports PDF and TXT files.
- **Automatic Summarization:** Get a concise summary (100–150 words) of your document.
- **Ask Anything:** Query your document and receive context-aware answers with justifications.
- **Challenge Me:** Take a quiz with logic-based questions generated from your document, and download your results as CSV.
- **Modern UI:** Clean, responsive, and visually appealing interface.
- **Offline-First:** All processing is done locally—your data stays private.

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Shubhroger482/Smart_Assistant.git
   cd Smart_Assistant
   ```

2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # Or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Start the app:**
   ```sh
   streamlit run main.py
   ```
2. **Open your browser:**
   - Go to the URL shown in the terminal (usually http://localhost:8501)
3. **Upload a document:**
   - Use the sidebar to upload a PDF or TXT file.
4. **Interact:**
   - View the summary, ask questions, or take the quiz challenge.

## Architecture / Reasoning Flow

```
User Uploads Document (PDF/TXT)
        |
        v
[PDF Parser / TXT Reader]
        |
        v
[Chunker]  <--- splits text into manageable pieces
        |
        v
[Summarizer]  <--- generates summary for display
        |
        v
[User selects mode: Ask Anything / Challenge Me]
        |                                 |
        |                                 |
        v                                 v
[Q&A Module]                      [Quiz Generator]
        |                                 |
        v                                 v
[Answer + Justification]         [Questions + Evaluation]
        |                                 |
        v                                 v
[Display in UI]                  [Score + Download Results]
```

- **main.py**: Streamlit UI, session state, and user interaction logic
- **utils/pdf_parser.py**: Extracts text from PDF files
- **utils/chunker.py**: Splits text into chunks for processing
- **backend/summarizer.py**: Generates summaries from text
- **backend/local_qa.py**: Answers user queries based on document content
- **backend/local_evaluator.py**: Generates quiz questions and evaluates answers

## Project Structure

```
main.py
requirements.txt
backend/
    local_evaluator.py
    local_qa.py
    summarizer.py
utils/
    chunker.py
    pdf_parser.py
```

## Dependencies
- streamlit
- python-dotenv
- pandas
- (See `requirements.txt` for full list)
- ---

**Made with ❤️ by Shubham Yadav (Shubhroger482)**
