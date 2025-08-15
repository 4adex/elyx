"""
Medical Conversation Agent using LangGraph
This agent orchestrates a conversation between a patient and doctor subagent.
"""

import asyncio
from typing import Dict, Any, List, TypedDict, Literal
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel


class ConversationStatus(Enum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_turn: Literal["patient", "doctor"]
    conversation_history: List[str]
    patient_query: str
    doctor_diagnosis: str
    status: ConversationStatus
    resolved: bool
    turn_count: int


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
        You are a patient seeking medical advice. You have a specific health concern or symptoms.
        
        Guidelines:
        - Be specific about your symptoms
        - Provide relevant medical history when asked
        - Ask follow-up questions if you need clarification
        - Be honest about your concerns and symptoms
        - Don't provide medical advice - you're seeking it
        - Keep responses conversational and natural
        - If the doctor provides a diagnosis and treatment plan that addresses your concerns, acknowledge it
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
        conversation_context = "\n".join(state["conversation_history"][-5:])
        
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
        You are a professional medical doctor providing consultation to a patient.
        
        Guidelines:
        - Ask relevant follow-up questions to understand symptoms better
        - Provide professional medical advice based on symptoms described
        - Suggest appropriate treatments or next steps
        - Be empathetic and professional
        - When you have enough information and have provided a comprehensive diagnosis and treatment plan, you can mark the consultation as resolved
        - To mark as resolved, end your response with "CONSULTATION_RESOLVED" on a new line
        - Only mark as resolved when you've provided a complete diagnosis and treatment recommendation
        
        Remember: This is for educational/demonstration purposes. Always recommend consulting with a real healthcare professional for actual medical concerns.
        """
        
        super().__init__(
            name="Doctor",
            role="doctor", 
            llm=llm,
            system_prompt=system_prompt
        )
    
    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate doctor response based on conversation state"""
        
        conversation_context = "\n".join(state["conversation_history"][-5:])
        
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
        
        # Check if doctor marked consultation as resolved
        resolved = "CONSULTATION_RESOLVED" in response.content
        
        return {
            "response": response.content.replace("CONSULTATION_RESOLVED", "").strip(),
            "agent": "doctor",
            "resolved": resolved
        }


class MedicalConversationAgent:
    """Main agent that orchestrates the conversation between patient and doctor"""
    
    def __init__(self, llm_model: str = None, temperature: float = None):
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
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("patient_turn", self._patient_node)
        workflow.add_node("doctor_turn", self._doctor_node)
        workflow.add_node("check_resolution", self._check_resolution_node)
        
        # Define the flow
        workflow.set_entry_point("patient_turn")
        
        # After patient turn, go to doctor
        workflow.add_edge("patient_turn", "doctor_turn")
        
        # After doctor turn, check if resolved
        workflow.add_edge("doctor_turn", "check_resolution")
        
        # From check_resolution, either end or continue conversation
        workflow.add_conditional_edges(
            "check_resolution",
            self._should_continue,
            {
                "continue": "patient_turn",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _patient_node(self, state: AgentState) -> AgentState:
        """Patient agent node"""
        print(f"\n--- Patient Turn (Turn {state['turn_count']}) ---")
        
        response_data = await self.patient_agent.respond(state)
        response = response_data["response"]
        
        print(f"Patient: {response}")
        
        # Update state
        state["conversation_history"].append(f"Patient: {response}")
        state["current_turn"] = "doctor"
        
        if state["turn_count"] == 0:
            state["patient_query"] = response
        
        return state
    
    async def _doctor_node(self, state: AgentState) -> AgentState:
        """Doctor agent node"""
        print(f"\n--- Doctor Turn (Turn {state['turn_count']}) ---")
        
        response_data = await self.doctor_agent.respond(state)
        response = response_data["response"]
        resolved = response_data.get("resolved", False)
        
        print(f"Doctor: {response}")
        
        if resolved:
            print("\n🏥 Doctor has marked the consultation as resolved!")
        
        # Update state
        state["conversation_history"].append(f"Doctor: {response}")
        state["current_turn"] = "patient"
        state["resolved"] = resolved
        state["turn_count"] += 1
        
        if resolved:
            state["status"] = ConversationStatus.RESOLVED
            state["doctor_diagnosis"] = response
        
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
            turn_count=0
        )
        
        # If initial query provided, add it to conversation
        if initial_patient_query:
            initial_state["patient_query"] = initial_patient_query
            initial_state["conversation_history"].append(f"Patient: {initial_patient_query}")
            print(f"Patient: {initial_patient_query}")
        
        try:
            # Run the conversation
            final_state = await self.graph.ainvoke(initial_state)
            
            print("\n" + "=" * 50)
            print("🏥 Medical Consultation Complete")
            
            return {
                "status": "completed",
                "conversation_history": final_state["conversation_history"],
                "patient_query": final_state["patient_query"],
                "doctor_diagnosis": final_state["doctor_diagnosis"],
                "resolved": final_state["resolved"],
                "total_turns": final_state["turn_count"]
            }
            
        except Exception as e:
            print(f"Error during conversation: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


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
