"""
Enhanced Orchestration with API Key Management and Rate Limiting
This file enhances the original orchestation.py with intelligent key rotation and rate limit handling.
"""

from typing import Dict, Any, List, TypedDict, Literal
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import os
import asyncio
import time

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from .conversation_logger import ConversationLogger
from .key_manager import APIKeyManager, ensure_key_manager_setup

load_dotenv()


class ConversationStatus(Enum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_turn: Literal["patient", "team"]
    conversation_history: List[Dict[str, Any]]
    patient_query: str
    status: ConversationStatus
    resolved: bool
    turn_count: int
    current_response: str
    current_agent: str
    control_agent: str  # Which team agent currently has control


@dataclass
class MedicalAgent:
    name: str
    role: str
    llm: BaseChatModel
    system_prompt: str

    def __init__(self, name: str, role: str, llm: BaseChatModel, system_prompt: str):
        # Add Elyx context and standard guidelines for all agents
        elyx_context = """
        You are part of Elyx - building the future of healthcare. Key principles:
        - We collaborate with members (not patients) over a long-term journey, typically a full year
        - Our goal is maximizing healthy years through prevention and personalized optimization
        - We don't just treat illnesses; we drive better health outcomes throughout life
        - Each member has a dynamic, personalized health plan that evolves with their journey
        - All interactions contribute to a comprehensive understanding of the member's goals, medical history, and health history
        - Everything a member does with Elyx (medical regimes, health plans, therapies) is part of their unique journey

        Here you are dealing with a patient, and this is just a thread of one of queries he asks in 4-5 times a week. So try to resolve it as soon as possible.
        """
        common_guidelines = """
        Common guidelines for all medical team members:
        - **VERY IMPORTANT** Keep responses focused and concise (2-3 lines maximum).
        - Frame advice in context of their personalized health plan and goals
        - If you think that the user intent has been decently resolved then just use the CONVERSATION_RESOLVED, in next line to end the conversation.
        
        You are consulting with Rohan Patel, a 46-year-old Regional Head of Sales at a FinTech company.
            He is based in Singapore but travels frequently to the UK, US, South Korea, and Jakarta.
            Married with two young children; has a supportive wife.
            He is highly analytical, time-conscious, and health-focused due to family history concerns.
            Prefers being addressed as Rohan.
            Keep your explanations clear, concise, and data-driven, while respecting his busy schedule.
        """

        worker_guidelines = """
        - Your job is to always do your part and transfer control back to ruby as soon as possible, use TRANSFER:RUBY on a new line to hand control back to Ruby.
        """
        if role.lower() == "patient":
            self.system_prompt = system_prompt.strip()
        elif role.lower() != "concierge":
            self.system_prompt = f"{elyx_context}\n\n{system_prompt.strip()}\n\n{common_guidelines}\n{worker_guidelines}"
        else:
            self.system_prompt = f"{elyx_context}\n\n{system_prompt.strip()}\n\n{common_guidelines}"
        self.name = name
        self.role = role
        self.llm = llm

    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Default respond - can be overridden by subclasses"""
        conversation_context = "\n".join([
            f"{msg['sender']}: {msg['message']}" for msg in state["conversation_history"][-12:]
        ])
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Recent conversation context (last messages):\n{conversation_context}\nPlease respond as {self.name} ({self.role}). ")
        ]
        response = await self.llm.ainvoke(messages)

        return {"response": response, "agent": self.name.lower()}


# --- Team agents ---
class PatientAgent(MedicalAgent):
    """Patient subagent representing Rohan Patel with a persisted profile in the system prompt."""
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Rohan Patel (Preferred name: Rohan). Member snapshot:
        - Preferred name: Rohan Patel
        - Date of birth: 12 March 1979 | Age: 46 | Gender: Male
        - Primary residence & frequent travel hubs: Singapore; frequently travels to UK, US, South Korea, Jakarta
        - Occupation: Regional Head of Sales for a FinTech company (frequent international travel, high-stress)

        Core outcomes & timelines:
        - Reduce risk of heart disease by maintaining healthy cholesterol and blood pressure levels by December 2026.
        - Enhance cognitive function and focus by June 2026.
        - Implement annual full-body health screenings starting November 2025.

        Behavioural & psychosocial insights:
        - Personality: Analytical, driven, values efficiency and evidence-based approaches.
        - Highly motivated but time-constrained; prefers executive summaries with clear recommendations and access to granular data on request.

        Tech stack & data feeds:
        - Uses Garmin for runs (willing to share Garmin data and other wearables).
        - Prefers monthly consolidated health reports and quarterly deep dives.

        Service & communication preferences:
        - Non-urgent responses within 24-48 hours; urgent concerns: contact PA immediately.
        - Language: English. Cultural note: Indian background; no special religious constraints.

        Response style rules:
        - Always reply in first person as Rohan Patel. Keep messages short and conversational (2-3 lines).
        - When asked for data, prefer executive-summary style with an offer to provide granular data.
        """
        super().__init__(name="Patient", role="patient", llm=llm, system_prompt=system_prompt)

    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate patient response using the stored profile. Keep replies to 2-3 lines and honor PA/escalation rules."""
        conversation_context = "".join([f"{msg['sender']}: {msg.get('message','')}" for msg in state["conversation_history"][-6:]])
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Recent conversation context (last messages): {conversation_context} You are Rohan Patel. Respond briefly (2-3 lines).")
        ]
        response = await self.llm.ainvoke(messages)
        return {"response": response.content.strip(), "agent": "patient"}


class ConciergeAgent(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Ruby, the Concierge / Orchestrator. Empathetic, organized, and proactive.
        - Always open the team's turn with a short 1-2 line message.
        - Announce which specialist will take the lead next by setting CONTROL:<agent_name>
        - You are currently responding to a small query asked by the user, after immediate query has been answered well, use CONSULTATION_RESOLVED on next line to END conversations.

        You need to choose from the specialist agents which do the following:
        - **Dr. Warren (Medical Strategist):** Use when the issue involves diagnosis, lab interpretation, or overall medical direction.  
        - **Advik (Performance Scientist):** Use when the issue involves wearable data, HRV, sleep, recovery, or stress patterns.  
        - **Carla (Nutritionist):** Use when the issue involves diet, supplements, meal planning, or food tracking.  
        - **Rachel (Physiotherapist):** Use when the issue involves injuries, exercise form, rehab, mobility, or physical training.  
        - **Neel (Concierge Lead):** Use when escalation, strategic review, or big-picture alignment with long-term goals is needed.  
        - Just respond with your message if you think that no specialist is needed. Otherwise pick the best specialist to respond next from: dr_warren, advik, carla, rachel, neel. Output two short lines: your message then on its own line CONTROL:<agent>.
        - **IMPORTANT** - Only use these specific words to transfer control, CONTROL:dr_warren, CONTROL:advik, CONTROL:carla, CONTROL:rachel, CONTROL:neel
        """

        super().__init__(name="Ruby", role="concierge", llm=llm, system_prompt=system_prompt)

    async def respond(self, state: AgentState) -> Dict[str, Any]:
        # Build a short assistant message and decide who should take control.
        # If state already contains a desired control_agent, Ruby should confirm and not override unless she chooses to.
        desired = state.get("control_agent")
        conversation_context = "\n".join([f"{m['sender']}: {m['message']}" for m in state["conversation_history"][-6:]])
        human_instructions = f"Context:\n{conversation_context}"
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_instructions)
        ]
        response = await self.llm.ainvoke(messages)
        text = response.content.strip()

        return {"response": text, "agent": "ruby"}


class MedicalStrategist(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Dr. Warren, the Medical Strategist. Authoritative, precise, scientific.
        - Give a short clinical take and if appropriate recommend tests or a diagnosis.
        """
        super().__init__(name="Dr_Warren", role="medical_strategist", llm=llm, system_prompt=system_prompt)


class PerformanceScientist(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Advik, the Performance Scientist. Analytical and pattern-oriented.
        - Focus on wearable data, HRV, sleep trends, and experiments.
        """
        super().__init__(name="Advik", role="performance_scientist", llm=llm, system_prompt=system_prompt)

class Nutritionist(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Carla, the Nutritionist. Practical and behavioral.
        - Give short nutrition guidance and actionable steps.
        """
        super().__init__(name="Carla", role="nutritionist", llm=llm, system_prompt=system_prompt)

class Physiotherapist(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Rachel, the PT / Physiotherapist. Direct and encouraging.
        - Provide short mobility/rehab or exercise cues.
        """
        
        super().__init__(name="Rachel", role="physio", llm=llm, system_prompt=system_prompt)

class ConciergeLead(MedicalAgent):
    def __init__(self, llm: BaseChatModel):
        system_prompt = """
        You are Neel, the Concierge Lead. Strategic and reassuring.
        - Step in for high-level summaries or to de-escalate.
        """
        super().__init__(name="Neel", role="concierge_lead", llm=llm, system_prompt=system_prompt)

# --- Control enum ---
class ControlAgent(Enum):
    RUBY = "ruby"
    DR_WARREN = "dr_warren"
    ADVIK = "advik"
    CARLA = "carla"
    RACHEL = "rachel"
    NEEL = "neel"


class EnhancedLLMWrapper:
    """
    Wrapper around ChatGroq that handles API key rotation and rate limiting.
    """
    
    def __init__(self, model: str, temperature: float, key_manager: APIKeyManager):
        self.model = model
        self.temperature = temperature
        self.key_manager = key_manager
        self.current_llm = None
        self.current_key = None
        self.current_key_nickname = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM with current week's API key."""
        try:
            self.current_key, self.current_key_nickname = self.key_manager.get_current_week_key()
            print(f"🔑 Using API key: {self.current_key_nickname} for this week")
            
            # Set the API key in environment
            os.environ["GROQ_API_KEY"] = self.current_key
            
            self.current_llm = ChatGroq(
                model=self.model, 
                temperature=self.temperature,
                groq_api_key=self.current_key
            )
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            raise
    
    async def ainvoke(self, messages, max_retries: int = 3):
        """
        Invoke LLM with automatic retry and key rotation on rate limits.
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = await self.current_llm.ainvoke(messages)
                end_time = time.time()
                
                # Record successful usage
                self.key_manager.record_api_usage(
                    self.current_key, 
                    success=True
                )
                
                print(f"✅ API call successful with {self.current_key_nickname} (took {end_time - start_time:.2f}s)")
                return response
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Record failed usage
                self.key_manager.record_api_usage(
                    self.current_key, 
                    success=False, 
                    error_message=str(e)
                )
                
                # Check if it's a rate limit error
                if "rate limit" in error_message or "429" in error_message:
                    print(f"🚫 Rate limit hit on {self.current_key_nickname} (attempt {attempt + 1})")
                    
                    if attempt < max_retries - 1:
                        try:
                            # Try to rotate to next available key
                            self.current_key, self.current_key_nickname = self.key_manager.rotate_to_next_key(self.current_key)
                            print(f"🔄 Switched to {self.current_key_nickname}")
                            
                            # Update environment and create new LLM instance
                            os.environ["GROQ_API_KEY"] = self.current_key
                            self.current_llm = ChatGroq(
                                model=self.model, 
                                temperature=self.temperature,
                                groq_api_key=self.current_key
                            )
                            
                            # Wait a bit before retrying
                            await asyncio.sleep(2)
                            continue
                            
                        except Exception as rotation_error:
                            print(f"❌ Could not rotate to next key: {rotation_error}")
                            if attempt == max_retries - 1:
                                raise e
                    else:
                        # No more retries
                        raise e
                else:
                    # Non-rate-limit error
                    print(f"❌ API error on {self.current_key_nickname}: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed to get response after {max_retries} attempts")


class OrchestatedAgent:
    def __init__(self, llm_model: str = None, temperature: float = None, log_dir: str = "logs"):
        # Initialize key manager first
        self.key_manager = ensure_key_manager_setup()
        
        try:
            from .config import Config
            Config.validate()
            model = llm_model or Config.DEFAULT_MODEL
            temp = temperature or Config.TEMPERATURE
        except Exception:
            model = llm_model or "llama3-8b-8192"
            temp = temperature or 0.7

        # Use enhanced LLM wrapper instead of direct ChatGroq
        self.llm = EnhancedLLMWrapper(model, temp, self.key_manager)

        # Team agents
        self.ruby = ConciergeAgent(self.llm)
        self.dr_warren = MedicalStrategist(self.llm)
        self.advik = PerformanceScientist(self.llm)
        self.carla = Nutritionist(self.llm)
        self.rachel = Physiotherapist(self.llm)
        self.neel = ConciergeLead(self.llm)
        # Patient agent wired into the swarm loop with Rohan's profile embedded in the system prompt
        self.patient_agent = PatientAgent(self.llm)

        self.logger = ConversationLogger(log_dir)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("ruby_node", self._ruby_node)
        workflow.add_node("format_ruby", self._format_ruby_message)
        workflow.add_node("control_dispatch", self._control_dispatch_node)
        workflow.add_node("agent_node", self._agent_node)
        workflow.add_node("format_agent", self._format_agent_message)
        workflow.add_node("patient_node", self._patient_node)
        workflow.add_node("format_patient", self._format_patient_message)
        workflow.add_node("check_resolution", self._check_resolution_node)

        # Flow with conditional branching for Ruby control
        workflow.set_entry_point("ruby_node")
        workflow.add_edge("ruby_node", "format_ruby")
        workflow.add_edge("format_ruby", "control_dispatch")
        
        # Add conditional edges from control_dispatch
        # TODO: Here is control dispatch only we will check for the resolution in the ruby's message if it exists
        workflow.add_conditional_edges(
            "control_dispatch",
            self._control_dispatch,
            {
                "ruby_flow": "patient_node",  # If control is ruby, skip agent node and go directly to patient
                "agent_flow": "agent_node" ,   # Otherwise continue through agent node
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "format_agent",
            self._should_continue_to_patient,
            {
                "end" : END,
                "continue": "patient_node"
            }
        )
        
        workflow.add_edge("agent_node", "format_agent")
        workflow.add_edge("patient_node", "format_patient")
        workflow.add_edge("format_patient", "check_resolution")

        workflow.add_conditional_edges(
            "check_resolution",
            self._should_continue,
            {
                "continue_agent": "control_dispatch",  # -> agent_node next (control_dispatch sets current_agent)
                "continue_ruby": "ruby_node",
                "end": END
            }
        )

        return workflow.compile()

    def _should_continue_to_patient(self, state: AgentState) -> str:
        MAX_TURNS = 4
        if state.get("turn_count", 0) >= MAX_TURNS:
            print(f"[guard] max turns {MAX_TURNS} reached -> ending conversation")
            state["resolved"] = True
            state["status"] = ConversationStatus.RESOLVED
            return "end"
        
        if state["resolved"]:
            return "end"
        
        return "continue"

    def _should_continue(self, state: AgentState) -> str:

        # Existing resolution check
        if state.get("status") == ConversationStatus.RESOLVED or state.get("resolved"):
            return "end"

        control = (state.get("control_agent") or "").lower()
        if control and control != ControlAgent.RUBY.value:
            # keep the current team agent in control (skip Ruby) — go via control_dispatch so agent_node sees correct current_agent
            return "continue_agent"

        return "continue_ruby"
    
    def _control_dispatch(self, state: AgentState) -> str:
        if state.get("status") == ConversationStatus.RESOLVED or state.get("resolved"):
            return "end"
        elif state.get("control_agent") == ControlAgent.RUBY.value:
            return "ruby_flow"
        else:
            return "agent_flow"


    # --- Nodes implementations ---
    async def _ruby_node(self, state: AgentState) -> AgentState:
        print(f"\n--- Ruby (Orchestrator) Turn (Turn {state['turn_count']}) ---")
        # Always have Ruby respond first in the team turn
        response_data = await self.ruby.respond(state)
        text = response_data["response"]

        state["current_response"] = text
        state["current_agent"] = "ruby"
        return state

    async def _format_ruby_message(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        
        resolved = any("CONSULTATION_RESOLVED" in line for line in response.splitlines())
        if resolved:
            state["resolved"] = True
            state["status"] = ConversationStatus.RESOLVED

        # Try to parse CONTROL:<agent> from ruby message; if present, keep it
        control = None
        for line in response.splitlines():
            if "CONTROL:" in line:
                control = line.split("CONTROL:", 1)[1].strip().lower()
                # Remove dots, commas from the control text
                control = control.replace(".", "").replace(",", "")
                break
        if control:
            state["control_agent"] = control
        
        lines = [line for line in response.splitlines() if "CONTROL:" not in line]
        response = "\n".join(lines).strip()

        structured = {"sender": "ruby", "role": "concierge", "message": response}
        print(f"Ruby: {structured['message']}")
        state["conversation_history"].append(structured)
        state["current_turn"] = "team"
        return state

    async def _control_dispatch_node(self, state: AgentState) -> AgentState:
        # Decide which agent should act now (control_agent must be set by Ruby)
        control = state.get("control_agent") or ControlAgent.DR_WARREN.value
        # normalize aliases
        control = control.lower()
        
        state["current_agent"] = control
        return state

    async def _agent_node(self, state: AgentState) -> AgentState:
        control = state.get("current_agent")
        print(f"\n--- Team Agent Turn: {control} (Turn {state['turn_count']}) ---")
        #TODO: It should not be possible that agent is something else, add a fallback here
        agent_obj = {
            ControlAgent.DR_WARREN.value: self.dr_warren,
            ControlAgent.ADVIK.value: self.advik,
            ControlAgent.CARLA.value: self.carla,
            ControlAgent.RACHEL.value: self.rachel,
            ControlAgent.NEEL.value: self.neel
        }.get(control)

        # Fallback if not valid
        if not agent_obj:
            state["status"] = "resolved"
            state["resolved"] = True
            return state

        response_data = await agent_obj.respond(state)
        text = response_data.get("response", "")
            
        state["current_response"] = text
        return state

    async def _format_agent_message(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        agent = state.get("current_agent")
        text = response.content.strip()

        for part in text.splitlines():
            if "TRANSFER:" in part:
                state["control_agent"] = "ruby"

        resolved = any("CONSULTATION_RESOLVED" in line for line in text.splitlines())
        if resolved:
            state["resolved"] = True
            state["status"] = ConversationStatus.RESOLVED

        lines = [line for line in text.splitlines() if not (("CONTROL:" in line) or ("TRANSFER:" in line) or ("CONSULTATION_RESOLVED" in line))]
        text = "\n".join(lines).strip()

        #TODO: Manage reasoning for agents

        structured = {"sender": agent, "role": agent, "message": text}

        print(f"{agent.capitalize()}: {structured['message']}")

        state["conversation_history"].append(structured)
        state["current_turn"] = "patient"
        # Increase turn count after a full team->patient cycle; we'll increment when doctor responds
        return state

    # Keep existing patient node and formatting with minor tweaks
    async def _patient_node(self, state: AgentState) -> AgentState:
        print(f"\n--- Patient Turn (Turn {state['turn_count']}) ---")
        # If the patient query is present and it's the first patient turn, use it; otherwise generate
        if state["turn_count"] == 0 and state.get("patient_query"):
            response = state["patient_query"]
        else:
            # generate a patient response via the wired PatientAgent (Rohan Patel)
            resp = await self.patient_agent.respond(state)
            response = resp.get("response", "")

        state["current_response"] = response
        state["current_agent"] = "patient"
        return state

    async def _format_patient_message(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        structured = {"sender": "patient", "role": "patient", "message": response}
        print(f"Patient: {structured['message']}")
        state["conversation_history"].append(structured)
        # increment turn count each full cycle (team+patient)

        #TODO: Patient should also be able to end the conversation
        state["turn_count"] += 1
        state["current_turn"] = "team"
        return state

    async def _check_resolution_node(self, state: AgentState) -> AgentState:
        # No-op; resolution handled when agent sets state['resolved']
        return state

    async def start_conversation(self, initial_patient_query: str = None) -> Dict[str, Any]:
        print("🏥 Starting Enhanced Swarm Medical Consultation with Load Balancing")
        print("=" * 60)

        initial_state = AgentState(
            messages=[],
            current_turn="team",
            conversation_history=[],
            patient_query="",
            status=ConversationStatus.ONGOING,
            resolved=False,
            turn_count=0,
            current_response="",
            current_agent="",
            control_agent=ControlAgent.DR_WARREN.value
        )

        if initial_patient_query:
            initial_state["patient_query"] = initial_patient_query
            initial_message = {"sender": "patient", "role": "patient", "message": initial_patient_query}
            initial_state["conversation_history"].append(initial_message)
            print(f"Patient (initial): {initial_patient_query}")
            initial_state["turn_count"] = 1
            
        try:
            final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 100})

            print("\n" + "=" * 60)
            print("🏥 Enhanced Swarm Medical Consultation Complete")

            conversation_data = {
                "status": "completed",
                "conversation_history": final_state["conversation_history"],
                "patient_query": final_state["patient_query"],
                "resolved": final_state["resolved"],
                "total_turns": final_state["turn_count"],
                "api_key_used": self.llm.current_key_nickname,
                "key_usage_summary": self.key_manager.get_usage_summary()
            }

            log_file = self.logger.log_conversation(conversation_data)
            print(f"\n💾 Conversation logged to: {log_file}")
            
            # Print key usage summary
            print(f"\n🔑 API Key Used: {self.llm.current_key_nickname}")
            
            return conversation_data

        except Exception as e:
            error_data = {
                "status": "error",
                "error": str(e),
                "conversation_history": initial_state["conversation_history"],
                "total_turns": initial_state["turn_count"],
                "api_key_used": getattr(self.llm, 'current_key_nickname', 'unknown'),
                "key_usage_summary": self.key_manager.get_usage_summary()
            }
            import traceback
            error_traceback = traceback.format_exc()
            log_file = self.logger.log_conversation(error_data)
            print(f"\n💾 Error state logged to: {log_file}")
            print(f"Error during conversation: {str(e)}")
            print("Traceback:")
            print(error_traceback)
            return error_data

    def get_key_manager_status(self) -> Dict[str, Any]:
        """Get current status of the key manager."""
        return self.key_manager.get_usage_summary()

    def print_key_status(self) -> None:
        """Print current key status."""
        self.key_manager.print_usage_summary()
