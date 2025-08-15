"""Message formatter for structured outputs in medical conversation."""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

class MessageFormatter:
    """Formats messages into structured output"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.doctor_system_prompt = """
        You are a message formatter for a medical conversation.
        Convert the given message into a structured format with these fields:
        - message: The actual response to the patient (excluding any internal medical reasoning)
        - reasoning: Your medical rationale for the response, observations about symptoms, and thought process
        
        Format your response as a Python dict like:
        {
            "message": "...",  # The actual response to patient
            "reasoning": "..."  # Your medical reasoning/rationale
        }
        
        Extract or infer the reasoning from the message context. If there's no explicit reasoning,
        provide a brief medical rationale for the response given.
        """
        
        self.patient_system_prompt = """
        You are a message formatter for a medical conversation.
        Convert the given message into a structured format with sender and message fields.
        
        Do not change the content or meaning of the message, only structure it.
        """
    
    def format_doctor_message(self, message: str) -> Dict[str, Any]:
        """Format doctor's message into structured output"""
        # Split message by REASONING: if present
        parts = message.split("\n\nREASONING:")
        
        if len(parts) == 2:
            # If REASONING: is found, use the split parts
            patient_message = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            # If no REASONING: found, keep whole message
            patient_message = message.strip()
            reasoning = ""
            
        return {
            "sender": "doctor",
            "message": patient_message,
            "reasoning": reasoning
        }
    
    def format_patient_message(self, message: str) -> Dict[str, Any]:
        """Format patient's message into structured output"""
        messages = [
            SystemMessage(content=self.patient_system_prompt),
            HumanMessage(content=f"Format this patient's message: {message}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            structured = eval(response.content)
            return {
                "sender": "patient",
                "message": structured["message"]
            }
        except:
            # Fallback to basic structure
            return {
                "sender": "patient",
                "message": message
            }
