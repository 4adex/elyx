"""
Simple runner script for the Medical Conversation Agent
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.medical_conversation_agent import MedicalConversationAgent
    from src.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install the required dependencies first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data with proper indentation and color if available"""
    try:
        # Try to use rich for colored output if available
        from rich import print_json
        return print_json(data)
    except ImportError:
        # Fallback to standard json formatting
        return json.dumps(data, indent=2, ensure_ascii=False)

def display_conversation_logs(log_dir: str = "logs", limit: int = 5):
    """Display recent conversation logs in JSON format"""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"\n❌ Log directory '{log_dir}' not found.")
        return
    
    log_files = sorted(log_path.glob("medical_conversation_*.json"), reverse=True)
    if not log_files:
        print(f"\n❌ No conversation logs found in {log_dir}")
        return
    
    print("\n📜 RECENT CONVERSATION LOGS")
    print("=" * 60)
    
    for log_file in log_files[:limit]:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                print(f"\n📝 Log File: {log_file.name}")
                print("-" * 60)
                print(format_json(log_data))
                print("\n" + "=" * 60)
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")


def print_banner():
    """Print a welcome banner"""
    print("🏥" + "=" * 60 + "🏥")
    print("     MEDICAL CONVERSATION AGENT - LANGGRAPH DEMO")
    print("🏥" + "=" * 60 + "🏥")
    print("\nThis agent simulates a conversation between:")
    print("👤 Patient Agent - Asks questions and describes symptoms")
    print("👨‍⚕️ Doctor Agent - Provides medical advice and diagnosis")
    print("\nThe conversation continues until the doctor marks it as resolved.")
    print("-" * 62)


async def run_interactive_mode(show_logs: bool = True):
    """Run the agent in interactive mode"""
    print_banner()
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize the agent
        agent = MedicalConversationAgent()
        
        print("\n🔧 Agent initialized successfully!")
        print("\nYou can either:")
        print("1. Enter a custom patient query")
        print("2. Press Enter to use a sample query")
        
        user_input = input("\nEnter patient query (or press Enter for sample): ").strip()
        
        # Use sample query if none provided
        if not user_input:
            sample_queries = [
                "I've been having persistent headaches for the past week, especially in the morning.",
                "I have a sore throat and difficulty swallowing that started 3 days ago.",
                "I've been experiencing chest pain when I exercise."
            ]
            user_input = sample_queries[0]
            print(f"Using sample query: {user_input}")
        
        print("\n🚀 Starting conversation...")
        
        # Run the conversation
        result = await agent.start_conversation(user_input)
        
        # Show conversation logs if requested
        if show_logs:
            display_conversation_logs()
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease check your .env file and ensure GROQ_API_KEY is set.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True


async def run_batch_mode():
    """Run the agent with multiple sample queries"""
    print_banner()
    
    try:
        Config.validate()
        agent = MedicalConversationAgent()
        
        sample_queries = [
            "I've been having severe headaches and dizziness for the past week.",
            "I have a persistent cough with some chest congestion for 5 days.",
            "I'm experiencing stomach pain after eating, especially spicy foods."
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n🔄 Running Conversation {i}/3")
            print("=" * 50)
            await agent.start_conversation(query)
        
        print("\n✅ Batch processing complete!")
        print("\n📜 Showing conversation logs:")
        display_conversation_logs(limit=3)  # Show logs for the batch conversations
        
    except Exception as e:
        print(f"\n❌ Error in batch mode: {e}")
        return False
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Conversation Agent Runner")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode with sample queries")
    parser.add_argument("--logs", action="store_true", help="Show conversation logs")
    parser.add_argument("--logs-only", action="store_true", help="Only display conversation logs without running new conversation")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing conversation logs")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent logs to display")
    
    args = parser.parse_args()
    
    if args.logs_only:
        display_conversation_logs(args.log_dir, args.limit)
    elif args.batch:
        asyncio.run(run_batch_mode())
    else:
        asyncio.run(run_interactive_mode(show_logs=args.logs))


if __name__ == "__main__":
    main()
