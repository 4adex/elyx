"""
Medical Conversation Agent (Swarm edition)
This file is an enhanced version of your original agent that implements
- Ruby (Concierge / Orchestrator) who always starts the turn
- A control enum that determines which specialist responds next
- Specialist subagents: Dr. Warren, Advik, Carla, Rachel, Neel
- Dynamic control transfer (agents may hand over control)
- Conversation loop: Ruby -> chosen specialist -> patient -> Ruby -> ...

Notes:
- This file keeps dependencies on MessageFormatter and ConversationLogger but formats team messages directly (so it works even if MessageFormatter lacks team-specific methods).
- Agents return small messages and may include a `transfer_to` field to request the next control.
- Any agent may mark the consultation resolved by including the token "CONSULTATION_RESOLVED" in their response; the orchestrator strips that and sets resolved.
"""

import asyncio
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from .conversation_logger import ConversationLogger


class ConversationStatus(Enum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_turn: Literal["patient", "team"]
    conversation_history: List[Dict[str, Any]]
    patient_query: str
    doctor_diagnosis: str
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
        """
        
        common_guidelines = """
        Common guidelines for all medical team members:
        - Keep responses focused and concise (2-3 lines maximum).
        - When your part is complete or another specialist would be more appropriate, use TRANSFER:RUBY on a new line to hand control back to Ruby.
        - Always consider the member's full journey and long-term health outcomes
        - Frame advice in context of their personalized health plan and goals
        """
        self.system_prompt = f"{elyx_context}\n\n{system_prompt.strip()}\n\n{common_guidelines}"
        self.name = name
        self.role = role
        self.llm = llm

    async def respond(self, state: AgentState) -> Dict[str, Any]:
        """Default respond - can be overridden by subclasses"""
        conversation_context = "\n".join([
            f"{msg['sender']}: {msg['message']}" for msg in state["conversation_history"][-6:]
        ])
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Recent conversation context (last messages):\n{conversation_context}\nPlease respond as {self.name} ({self.role}). ")
        ]
        response = await self.llm.ainvoke(messages)

        text = response.content
        resolved = "CONSULTATION_RESOLVED" in text
        transfer = None
        for part in text.splitlines():
            if "TRANSFER:" in part:
                transfer = "ruby"
                break
        
        # Clean response by removing control tokens
        lines = [line for line in text.splitlines() if not (("CONTROL:" in line) or ("TRANSFER:" in line) or ("CONSULTATION_RESOLVED" in line))]
        text = "\n".join(lines).strip()

        return {"response": text, "agent": self.name.lower(), "resolved": resolved, "transfer_to": transfer}


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
        - Announce which specialist will take the lead next by setting CONTROL:<agent_name> in your message (e.g. CONTROL:dr_warren).
        - Use plain language and confirm actions. Keep it tiny (1-2 lines).
        - If you think that good enough consulting has been done or if the conversation grew very long then just sum it up for patient and include CONSULTATION_RESOLVED on a new line to mark the end.
        """
        super().__init__(name="Ruby", role="concierge", llm=llm, system_prompt=system_prompt)

    async def respond(self, state: AgentState) -> Dict[str, Any]:
        # Build a short assistant message and decide who should take control.
        # If state already contains a desired control_agent, Ruby should confirm and not override unless she chooses to.
        desired = state.get("control_agent")
        conversation_context = "\n".join([f"{m['sender']}: {m['message']}" for m in state["conversation_history"][-6:]])
        human_instructions = f"Context:\n{conversation_context}\n\nIf control_agent is set to '{desired}', confirm and repeat CONTROL:{desired}. Otherwise pick the best specialist to respond next from: dr_warren, advik, carla, rachel, neel. Output two short lines: your message then on its own line CONTROL:<agent>"
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_instructions)
        ]
        response = await self.llm.ainvoke(messages)
        text = response.content.strip()

        # Try to parse CONTROL:<agent> from Ruby's message
        control = None
        for part in text.splitlines()[1:]:
            if "CONTROL:" in part:
                control = part.split("CONTROL:", 1)[1].strip().lower()
                break
        # Fallback: if desired set, use it
        if not control and desired:
            control = desired
        if not control:
            # default fallback
            control = "ruby"

        # Remove all of the lines with CONTROL in that
        lines = [line for line in text.splitlines() if "CONTROL:" not in line]
        text = "\n".join(lines).strip()

        return {"response": text, "agent": "ruby", "transfer_to": control}


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


class OrchestatedAgent:
    def __init__(self, llm_model: str = None, temperature: float = None, log_dir: str = "logs"):
        try:
            from .config import Config
            Config.validate()
            model = llm_model or Config.DEFAULT_MODEL
            temp = temperature or Config.TEMPERATURE
        except Exception:
            model = llm_model or "llama3-8b-8192"
            temp = temperature or 0.7

        self.llm = ChatGroq(model=model, temperature=temp)

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

        # Flow: Ruby -> format -> control dispatch -> agent -> format -> patient -> format -> check_resolution
        workflow.set_entry_point("ruby_node")
        workflow.add_edge("ruby_node", "format_ruby")
        workflow.add_edge("format_ruby", "control_dispatch")
        workflow.add_edge("control_dispatch", "agent_node")
        workflow.add_edge("agent_node", "format_agent")
        workflow.add_edge("format_agent", "patient_node")
        workflow.add_edge("patient_node", "format_patient")
        workflow.add_edge("format_patient", "check_resolution")

        # From check_resolution decide whether to go back to ruby_node or to the next agent (via control_dispatch), or end.
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


    def _should_continue(self, state: AgentState) -> str:
        """Decide next step after patient turn:
           - If resolved or turns >= limit -> end
           - If control_agent is set and not 'ruby' -> continue_agent (call control_dispatch -> agent_node)
           - Otherwise -> continue_ruby (Ruby gets the next team turn)
        """
        if state.get("resolved"):
            return "end"

        control = (state.get("control_agent") or "").lower()
        if control and control != ControlAgent.RUBY.value:
            # keep the current team agent in control (skip Ruby) — go via control_dispatch so agent_node sees correct current_agent
            return "continue_agent"

        return "continue_ruby"

    # --- Nodes implementations ---
    async def _ruby_node(self, state: AgentState) -> AgentState:
        print(f"\n--- Ruby (Orchestrator) Turn (Turn {state['turn_count']}) ---")
        # Always have Ruby respond first in the team turn
        response_data = await self.ruby.respond(state)
        text = response_data["response"]
        transfer_to = response_data.get("transfer_to")

        state["current_response"] = text
        state["current_agent"] = "ruby"
        if transfer_to:
            state["control_agent"] = transfer_to
        return state

    async def _format_ruby_message(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        # Try to parse CONTROL:<agent> from ruby message; if present, keep it
        control = None
        for line in response.splitlines():
            if "CONTROL:" in line:
                control = line.split("CONTROL:", 1)[1].strip().lower()
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
        agent_obj = {
            ControlAgent.DR_WARREN.value: self.dr_warren,
            ControlAgent.ADVIK.value: self.advik,
            ControlAgent.CARLA.value: self.carla,
            ControlAgent.RACHEL.value: self.rachel,
            ControlAgent.NEEL.value: self.neel
        }.get(control, self.ruby)

        response_data = await agent_obj.respond(state)
        text = response_data.get("response", "")
        resolved = response_data.get("resolved", False)
        transfer = response_data.get("transfer_to")

        # If agent asked to transfer control, record it so Ruby can confirm next loop.
        if transfer:
            # canonicalize common names
            state["control_agent"] = transfer
        state["current_response"] = text
        state["resolved"] = state.get("resolved") or resolved
        return state

    async def _format_agent_message(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        agent = state.get("current_agent")

        # Extract REASONING: block if present and keep it in structured data
        reasoning = None
        if "REASONING:" in response:
            parts = response.split("REASONING:", 1)
            main = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            main = response.strip()

        structured = {"sender": agent, "role": agent, "message": main}
        if reasoning:
            structured["reasoning"] = reasoning

        print(f"{agent.capitalize()}: {structured['message']}")
        if reasoning:
            print(f"Reasoning: {structured['reasoning']}")

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
        state["turn_count"] += 1
        state["current_turn"] = "team"
        return state

    async def _check_resolution_node(self, state: AgentState) -> AgentState:
        # No-op; resolution handled when agent sets state['resolved']
        return state

    async def start_conversation(self, initial_patient_query: str = None) -> Dict[str, Any]:
        print("🏥 Starting Swarm Medical Consultation")
        print("=" * 50)

        initial_state = AgentState(
            messages=[],
            current_turn="team",
            conversation_history=[],
            patient_query="",
            doctor_diagnosis="",
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

            print("\n" + "=" * 50)
            print("🏥 Swarm Medical Consultation Complete")

            conversation_data = {
                "status": "completed",
                "conversation_history": final_state["conversation_history"],
                "patient_query": final_state["patient_query"],
                "doctor_diagnosis": final_state.get("doctor_diagnosis", ""),
                "resolved": final_state["resolved"],
                "total_turns": final_state["turn_count"],
                "metadata": {
                    "model": getattr(self.llm, "model", None),
                    "temperature": getattr(self.llm, "temperature", None)
                }
            }

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
            log_file = self.logger.log_conversation(error_data)
            print(f"\n💾 Error state logged to: {log_file}")
            print(f"Error during conversation: {str(e)}")
            return error_data
