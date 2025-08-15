# Medical Conversation Agent with LangGraph

This project implements a LangGraph-based conversational agent that simulates a medical consultation between a patient and doctor. The system uses two specialized subagents that interact in turns until the consultation is resolved.

## 🏥 Features

- **Patient Agent**: Asks medical questions, describes symptoms, and seeks clarification
- **Doctor Agent**: Provides medical advice, asks follow-up questions, and can mark consultations as resolved
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
python run_agent.py
```

**Batch Mode** (test with multiple queries):
```bash
python run_agent.py --batch
```

## 📋 System Architecture

### Agent Flow

1. **Patient Turn**: Patient agent starts with a medical query or responds to doctor's questions
2. **Doctor Turn**: Doctor agent analyzes symptoms and provides advice
3. **Resolution Check**: System checks if doctor marked consultation as resolved
4. **Loop**: Continue conversation until resolved or max turns reached

### LangGraph Structure

```
Patient Turn → Doctor Turn → Check Resolution
     ↑              ↓              ↓
     ← ← ← Continue ← ← ← ← ← ← End (if resolved)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required |
| `DEFAULT_MODEL` | LLM model to use | `llama3-8b-8192` |
| `TEMPERATURE` | Model temperature | `0.7` |
| `MAX_TURNS` | Max conversation turns | `10` |

### Customization

You can modify the system prompts in `medical_conversation_agent.py`:

- **Patient Agent**: Lines 44-54 (patient behavior and guidelines)
- **Doctor Agent**: Lines 85-99 (medical advice and resolution criteria)

## 📁 Project Structure

```
elyx/
├── src/
│   ├── medical_conversation_agent.py  # Main LangGraph agent implementation
│   └── config.py                      # Configuration management
├── run_agent.py                       # Runner script with interactive/batch modes
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
└── README.md                          # This file
```

## 🔍 Example Conversation

```
🏥 Starting Medical Consultation

--- Patient Turn (Turn 0) ---
Patient: I've been having persistent headaches for the past week, especially in the morning. They seem to get worse when I stand up quickly.

--- Doctor Turn (Turn 0) ---
Doctor: I understand you're experiencing persistent headaches, particularly in the morning and when standing up quickly. This combination of symptoms is concerning and I'd like to gather more information to help determine the cause.

Can you tell me:
1. How would you rate the pain intensity on a scale of 1-10?
2. Are you experiencing any nausea, vomiting, or visual changes?
3. Have you had any recent changes in sleep, stress, or medications?
4. How much water do you typically drink per day?

The pattern of worsening when standing suggests we should consider blood pressure or hydration issues.

--- Patient Turn (Turn 1) ---
Patient: The pain is usually around 6-7 out of 10. I do feel slightly nauseous in the mornings, but no vomiting or vision problems. My sleep has been poor lately due to work stress, and I probably only drink 3-4 glasses of water daily.

--- Doctor Turn (Turn 1) ---
Doctor: Thank you for the detailed information. Based on your symptoms - morning headaches rated 6-7/10, mild nausea, poor sleep due to stress, and low water intake - I believe you're experiencing tension headaches with dehydration as a contributing factor.

**My Recommendations:**
1. **Hydration**: Increase water intake to 8-10 glasses daily
2. **Sleep hygiene**: Aim for 7-8 hours, establish a regular bedtime routine
3. **Stress management**: Try relaxation techniques, meditation, or light exercise
4. **Gradual position changes**: Stand up slowly to prevent orthostatic symptoms
5. **Pain management**: Over-the-counter NSAIDs (ibuprofen) can help with acute pain

**Follow-up**: If headaches persist after 1-2 weeks of these lifestyle changes, or if you develop severe symptoms like high fever, severe nausea, vision changes, or neck stiffness, please seek immediate medical attention.

This consultation addresses your concerns comprehensively. Please implement these recommendations and monitor your symptoms.

CONSULTATION_RESOLVED

🏥 Doctor has marked the consultation as resolved!

📊 Final Result: {'status': 'completed', 'resolved': True, 'total_turns': 2}
```

## 🛠 Advanced Usage

### Custom Patient Queries

```python
from src.medical_conversation_agent import MedicalConversationAgent

agent = MedicalConversationAgent()
result = await agent.start_conversation("I have been experiencing chest pain during exercise")
```

### Different LLM Models

```python
agent = MedicalConversationAgent(llm_model="llama3-70b-8192")
```

### Available Groq Models

- `llama3-8b-8192` - Fast and efficient (default)
- `llama3-70b-8192` - More capable, slower
- `mixtral-8x7b-32768` - Good balance of speed and capability
- `gemma-7b-it` - Google's Gemma model

## ⚠ Important Notes

- This is for **educational/demonstration purposes only**
- Always recommend consulting with real healthcare professionals for actual medical concerns
- The agent includes disclaimers in its responses about seeking professional medical advice
- Conversation is limited to prevent infinite loops (max 10 turns by default)

## 📚 Dependencies

- **LangGraph**: For workflow orchestration
- **LangChain**: For LLM integration and message handling
- **Groq**: For fast LLM inference with Llama and other models
- **python-dotenv**: For environment variable management

## 🔮 Future Enhancements

- Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- Conversation history persistence
- Medical knowledge base integration
- Multi-language support
- Web interface
- Voice integration

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the medical conversation agent!

## 📄 License

This project is for educational purposes. Please ensure you comply with your LLM provider's terms of service when using this code.
