"""Logger for medical conversation histories."""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

class ConversationLogger:
    """Handles logging of medical conversations to JSON files"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger with a directory for log files"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Log a complete conversation to a JSON file
        
        Args:
            conversation_data: Dictionary containing conversation history and metadata
            
        Returns:
            str: Path to the created log file
        """
        # Create a timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_conversation_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Prepare the log data with metadata
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": timestamp,
            **conversation_data
        }
        
        # Write to JSON file with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
        return filepath
