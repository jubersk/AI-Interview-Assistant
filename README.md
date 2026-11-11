
# AI Interview Assistant — Community Edition

**Version:** 3.7
**Author:** jbengineer

A locally running AI-powered interview preparation assistant built with Python, Streamlit, Ollama, and Faster-Whisper.

Designed to help professionals prepare for technical, leadership, and behavioral interviews using their own CVs and a local language model.

## Features

- Local AI interview assistance using Ollama
- CV-based interview answer generation
- Multiple CV support
- Custom job description context
- SHORT, DETAILED, CODE, and STAR answer modes
- Technical knowledge base using JSON
- Mock interviews with feedback
- Microphone-based transcription
- Local processing without a paid OpenAI API key

## Technology Stack

- Python
- Streamlit
- Ollama (Llama 3.2)
- Faster-Whisper
- pypdf
- NumPy and SciPy

## Prerequisites

- Python 3.13 (tested environment)
- Ollama installed and running
- Llama 3.2 model
- Microphone access for transcription features

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jubersk/AI-Interview-Assistant.git
cd AI-Interview-Assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and configure Ollama

Download Ollama from https://ollama.com.

Pull the model:

```bash
ollama pull llama3.2
```

Ensure the Ollama service is running.

### 5. Add your CV

Create a directory named `cvs` in the project root.

Place your PDF CV files inside the directory.

Personal CV files are excluded from Git tracking.

To use a master CV, name the PDF Master_CV.pdf and place it inside the cvs/ directory.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501.

## How to Use

1. Select a CV and interview role.
2. Optionally provide a job description.
3. Select SHORT, DETAILED, CODE, or STAR mode.
4. Enter an interview question.
5. Generate an AI-assisted answer.
6. Use Mock Interview Mode to practice and review feedback.

## Example Questions

- Explain your automation framework architecture.
- How do you handle flaky Selenium tests?
- What is the difference between HashMap and ConcurrentHashMap?
- How do you manage critical defects before production release?

## Privacy

The application uses a locally hosted Ollama model for answer generation.

CVs are stored locally and excluded from Git tracking.

Speech transcription may require downloading a model during initial setup.

Review external model downloads and dependencies before use.

## Disclaimer

This application is intended for interview preparation and practice. AI-generated answers should be reviewed for accuracy and adapted to your actual professional experience.

Copyright (c) 2026 jbengineer. All rights reserved.
