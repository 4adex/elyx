"""
Medical Conversation Agent Package
LangGraph-based conversational agent for medical consultations
"""

from .medical_conversation_agent import (
    MedicalConversationAgent,
    PatientAgent, 
    DoctorAgent,
    ConversationStatus,
    AgentState
)

from .config import Config

__version__ = "1.0.0"
__author__ = "Assistant"

__all__ = [
    "MedicalConversationAgent",
    "PatientAgent",
    "DoctorAgent", 
    "ConversationStatus",
    "AgentState",
    "Config"
]
