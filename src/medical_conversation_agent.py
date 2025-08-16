"""
Medical Conversation Agent using LangGraph
This agent orchestrates a conversation between a patient and doctor subagent.
"""

import asyncio
from typing import Dict, Any, List, TypedDict, Literal
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from .message_formatter import MessageFormatter
from .conversation_logger import ConversationLogger


class ConversationStatus(Enum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_turn: Literal["patient", "doctor"]
    conversation_history: List[Dict[str, Any]]  # List of structured messages
    patient_query: str
    doctor_diagnosis: str
    status: ConversationStatus
    resolved: bool
    turn_count: int
    current_response: str  # Temporary storage for current response
    current_agent: str  # Temporary storage for current agent


@dataclass
class MedicalAgent:
    """Base class for medical conversation agents"""
    name: str
    role: str
    llm: BaseChatModel
    system_prompt: str


class PatientAgent(MedicalAgent):
    """Patient subagent that asks medical questions and provides symptoms"""
    
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are a patient seeking medical advice. Act like a real person having a natural conversation.
        
        Guidelines:
        - Keep messages short and conversational, like texting with a doctor (2-3 sentences max)
        - Describe symptoms briefly and clearly
        - Ask questions naturally when you need clarification
        - Don't explain everything at once - let the doctor ask follow-up questions
        - Respond directly to the doctor's questions
        - Show natural concern but avoid being overly dramatic
        - Use everyday language, not medical terminology
        """
        
        super().__init__(
            name="Patient",
            role="patient",
            llm=llm,
            system_prompt=system_prompt
        )
    
    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate patient response based on conversation state"""
        
        # Build conversation context with last 5 messages
        conversation_context = "\n".join([
            f"{msg['sender']}: {msg['message']}" 
            for msg in state["conversation_history"][-5:]
        ])
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Recent conversation context (last 5 messages):
            {conversation_context}
            
            Current situation: You are {'starting the conversation with your medical concern' if state['turn_count'] == 0 else 'responding to the doctor'}
            
            Generate your response as the patient.
            """)
        ]
        
        response = await self.llm.ainvoke(messages)
        return {
            "response": response.content,
            "agent": "patient"
        }


class DoctorAgent(MedicalAgent):
    """Doctor subagent that provides medical advice and diagnosis"""
    
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are a friendly doctor having a natural conversation with a patient.
        
        Guidelines:
        - Keep messages brief and conversational (2-3 sentences max)
        - Ask one focused question at a time
        - Use simple, everyday language instead of medical jargon
        - Show empathy through brief, natural responses
        - Build the conversation gradually - don't try to diagnose immediately
        
        Structure your responses in two parts:
        1. Patient Message: Your actual response to the patient in simple terms
        2. Medical Reasoning: Add your medical thoughts/observations after "REASONING:" on a new line
        
        When ready to conclude:
        1. Give a brief diagnosis and simple treatment plan
        2. Include final reasoning
        3. End with "CONSULTATION_RESOLVED" on a new line
        
        Remember: This is for demonstration. Always include a reminder to see a real doctor.
        
        Example response:
        Hi there! That pain you described sounds like it could be tension-related. Can you tell me if it gets worse with movement?

        REASONING: Patient's symptoms suggest muscular tension. Need to rule out nerve involvement.
        """
        
        super().__init__(
            name="Doctor",
            role="doctor", 
            llm=llm,
            system_prompt=system_prompt
        )
    
    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate doctor response based on conversation state"""
        
        conversation_context = "\n".join([
            f"{msg['sender']}: {msg['message']}" 
            for msg in state["conversation_history"][-5:]
        ])
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Recent conversation context (last 5 messages):
            {conversation_context}
            
            The patient has just shared their concern/symptoms. Provide your medical assessment and advice.
            If you have enough information to provide a complete diagnosis and treatment plan, you may end with "CONSULTATION_RESOLVED" to end the consultation.
            """)
        ]
        
        response = await self.llm.ainvoke(messages)
        resolved = "CONSULTATION_RESOLVED" in response.content
        
        return {
            "response": response.content.replace("CONSULTATION_RESOLVED", "").strip(),
            "agent": "doctor",
            "resolved": resolved
        }


class MedicalConversationAgent:
    """Main agent that orchestrates the conversation between patient and doctor"""
    
    def __init__(self, llm_model: str = None, temperature: float = None, log_dir: str = "logs"):
        # Import config here to avoid circular imports
        try:
            from .config import Config
            Config.validate()
            model = llm_model or Config.DEFAULT_MODEL
            temp = temperature or Config.TEMPERATURE
        except ImportError:
            # Fallback if config module isn't available
            model = llm_model or "llama3-8b-8192"
            temp = temperature or 0.7
        
        self.llm = ChatGroq(model=model, temperature=temp)
        self.patient_agent = PatientAgent(self.llm)
        self.doctor_agent = DoctorAgent(self.llm)
        self.message_formatter = MessageFormatter(self.llm)
        self.logger = ConversationLogger(log_dir)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for agents and formatting
        workflow.add_node("patient_node", self._patient_node)
        workflow.add_node("format_patient", self._format_patient_message)
        workflow.add_node("doctor_node", self._doctor_node)
        workflow.add_node("format_doctor", self._format_doctor_message)
        workflow.add_node("check_resolution", self._check_resolution_node)
        
        # Define the flow
        workflow.set_entry_point("patient_node")
        
        # After patient response, format it
        workflow.add_edge("patient_node", "format_patient")
        
        # After formatting patient message, go to doctor
        workflow.add_edge("format_patient", "doctor_node")
        
        # After doctor response, format it
        workflow.add_edge("doctor_node", "format_doctor")
        
        # After formatting doctor message, check resolution
        workflow.add_edge("format_doctor", "check_resolution")
        
        # From check_resolution, either end or continue conversation
        workflow.add_conditional_edges(
            "check_resolution",
            self._should_continue,
            {
                "continue": "patient_node",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _patient_node(self, state: AgentState) -> AgentState:
        """Patient agent node"""
        print(f"\n--- Patient Turn (Turn {state['turn_count']}) ---")
        
        response_data = await self.patient_agent.respond(state)
        response = response_data["response"]
        
        # Store raw response for formatting
        state["current_response"] = response
        state["current_agent"] = "patient"
        
        if state["turn_count"] == 0:
            state["patient_query"] = response
        
        return state
    
    async def _format_patient_message(self, state: AgentState) -> AgentState:
        """Format patient message into structured output"""
        response = state["current_response"]
        structured_message = self.message_formatter.format_patient_message(response)
        
        print(f"Patient: {structured_message['message']}")
        
        state["conversation_history"].append(structured_message)
        state["current_turn"] = "doctor"
        
        return state
    
    async def _doctor_node(self, state: AgentState) -> AgentState:
        """Doctor agent node"""
        print(f"\n--- Doctor Turn (Turn {state['turn_count']}) ---")
        
        response_data = await self.doctor_agent.respond(state)
        response = response_data["response"]
        resolved = response_data.get("resolved", False)
        
        # Store raw response and resolved status for formatting
        state["current_response"] = response
        state["current_agent"] = "doctor"
        state["resolved"] = resolved
        
        if resolved:
            print("\n🏥 Doctor has marked the consultation as resolved!")
            state["status"] = ConversationStatus.RESOLVED
        
        return state
    
    async def _format_doctor_message(self, state: AgentState) -> AgentState:
        """Format doctor message into structured output"""
        response = state["current_response"]
        structured_message = self.message_formatter.format_doctor_message(response)
        
        print(f"Doctor: {structured_message['message']}")
        print(f"Reasoning: {structured_message['reasoning']}")
        
        state["conversation_history"].append(structured_message)
        state["current_turn"] = "patient"
        state["turn_count"] += 1
        
        if state["resolved"]:
            state["doctor_diagnosis"] = structured_message["message"]
        
        return state
    
    async def _check_resolution_node(self, state: AgentState) -> AgentState:
        """Check if the conversation should be resolved"""
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if conversation should continue or end"""
        if state["resolved"] or state["turn_count"] >= 10:  # Max 10 turns to prevent infinite loops
            return "end"
        return "continue"
    
    async def start_conversation(self, initial_patient_query: str = None) -> Dict[str, Any]:
        """Start the medical conversation"""
        
        print("🏥 Starting Medical Consultation")
        print("=" * 50)
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            current_turn="patient",
            conversation_history=[],
            patient_query="",
            doctor_diagnosis="",
            status=ConversationStatus.ONGOING,
            resolved=False,
            turn_count=0,
            current_response="",
            current_agent=""
        )
        
        # If initial query provided, add it to conversation
        if initial_patient_query:
            initial_state["patient_query"] = initial_patient_query
            # Add structured initial message
            initial_message = {
                "sender": "patient",
                "message": initial_patient_query
            }
            initial_state["conversation_history"].append(initial_message)
            print(f"Patient: {initial_patient_query}")
        
        try:
            # Run the conversation
            final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 100})
            
            print("\n" + "=" * 50)
            print("🏥 Medical Consultation Complete")
            
            # Prepare conversation data
            conversation_data = {
                "status": "completed",
                "conversation_history": final_state["conversation_history"],
                "patient_query": final_state["patient_query"],
                "doctor_diagnosis": final_state["doctor_diagnosis"],
                "resolved": final_state["resolved"],
                "total_turns": final_state["turn_count"],
                "metadata": {
                    "model": self.llm.model,
                    "temperature": self.llm.temperature
                }
            }
            
            # Log the conversation
            log_file = self.logger.log_conversation(conversation_data)
            print(f"\n💾 Conversation logged to: {log_file}")
            
            return conversation_data
            
        except Exception as e:
            error_data = {
                "status": "error",
                "error": str(e),
                "conversation_history": initial_state["conversation_history"],
                "total_turns": initial_state["turn_count"]
            }
            
            # Log error state
            log_file = self.logger.log_conversation(error_data)
            print(f"\n💾 Error state logged to: {log_file}")
            
            print(f"Error during conversation: {str(e)}")
            return error_data


# Example usage and testing
async def main():
    """Example usage of the Medical Conversation Agent"""
    
    # Initialize the agent
    agent = MedicalConversationAgent()
    
    # Example patient queries
    sample_queries = [
        "I've been having persistent headaches for the past week, especially in the morning. They seem to get worse when I stand up quickly.",
        "I have a sore throat and difficulty swallowing. It started 3 days ago and I also have a slight fever.",
        "I've been experiencing chest pain when I exercise. It's a sharp pain that goes away when I rest."
    ]
    
    # Start conversation with a sample query
    print("🚀 Running Medical Conversation Agent Demo")
    print("This demo shows a conversation between a patient and doctor agent.\n")
    
    result = await agent.start_conversation(sample_queries[0])
    
    print(f"\n📊 Final Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
