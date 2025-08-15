"""
Simple runner script for the Medical Conversation Agent
"""

import asyncio
import sys
import os

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


async def run_interactive_mode():
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
        
        # Display results
        print("\n📊 CONVERSATION SUMMARY")
        print("=" * 40)
        print(f"Status: {result['status']}")
        print(f"Total turns: {result.get('total_turns', 'N/A')}")
        print(f"Resolved: {result.get('resolved', False)}")
        
        if result['status'] == 'completed':
            print(f"\n🎯 Final Diagnosis/Advice:")
            print(f"{result.get('doctor_diagnosis', 'N/A')}")
        
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
            
            result = await agent.start_conversation(query)
            
            print(f"\nResult {i}: {result['status']}")
            if result['status'] == 'completed':
                print(f"Turns: {result['total_turns']}, Resolved: {result['resolved']}")
        
        print("\n✅ Batch processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error in batch mode: {e}")
        return False
    
    return True


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        asyncio.run(run_batch_mode())
    else:
        asyncio.run(run_interactive_mode())


if __name__ == "__main__":
    main()
