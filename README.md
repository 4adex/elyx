# Medical Conversation Agent with LangGraph

This project implements a LangGraph-based conversational agent that simulates a medical consultation between a patient and doctor. The system uses two specialized subagents that interact in turns until the consultation is resolved.

## 🏥 Features

- **Patient Agent**: Asks medical questions, describes symptoms, and seeks clarification
- **Multi Elyx Team Agent**: Provides medical advice, asks follow-up questions, and can mark consultations as resolved
- **LangGraph Orchestration**: Manages the conversation flow and turn-taking
- **Configurable**: Support for different LLM models and conversation parameters
- **Interactive & Batch Modes**: Run single conversations or test with multiple scenarios

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Run the Agent

**Interactive Mode** (default):
```bash
python3 run_enhanced_synthesizer.py
```
- Add your groq api keys for robin round key method to avoid rate limits.