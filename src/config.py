"""
Configuration module for Medical Conversation Agent
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the medical conversation agent"""
    
    # Groq API configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-8b-8192")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Conversation limits
    MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in your environment or .env file."
            )
        
        return True
