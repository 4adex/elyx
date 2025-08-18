"""
Enhanced 8-Month Patient Journey Synthesizer with API Key Load Balancing
This version includes intelligent API key rotation and rate limit handling for large-scale synthesis.
"""

import asyncio
import random
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# External LLM/orchestrator imports with enhanced key management
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .enhanced_orchestation import OrchestatedAgent
from .conversation_logger import ConversationLogger
from .journey_logger import JourneyLogger
from .key_manager import APIKeyManager, ensure_key_manager_setup


class WeekType(Enum):
    ONBOARDING = "onboarding"
    PHYSICAL_EXAM = "physical_exam"
    RESULTS_SHARING = "results_sharing"
    CHECKIN = "checkin"
    FORTNIGHTLY_CALL = "fortnightly_call"
    TRAVEL = "travel"
    BIOMARKER = "biomarker"
    EXERCISE_UPDATE = "exercise_update"
    PLAN_ADJUSTMENT = "plan_adjustment"
    NORMAL = "normal"


@dataclass
class WeekContext:
    week_number: int
    week_type: WeekType
    date_range: Tuple[datetime, datetime]
    additional_context: str
    travel_destination: str = None
    biomarker_focus: str = None
    plan_adherence: float = 0.5  # 50% default adherence


class EnhancedPatientJourneySynthesizer:
    def __init__(self, output_dir: str = "synthesized_journey"):
        # Initialize key manager first
        self.key_manager = ensure_key_manager_setup()
        
        # Initialize LLM with key management
        current_key, current_key_nickname = self.key_manager.get_current_week_key()
        print(f"🔑 Initializing synthesizer with key: {current_key_nickname}")
        
        self.llm = ChatGroq(
            model="llama3-8b-8192", 
            temperature=0.7,
            groq_api_key=current_key
        )
        
        # Use enhanced orchestrator
        self.orchestrator = OrchestatedAgent()
        self.conversation_logger = ConversationLogger(output_dir)
        self.journey_logger = JourneyLogger(output_dir)
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Patient profile for context
        self.patient_profile = {
            "name": "Rohan Patel",
            "age": 46,
            "occupation": "Regional Head of Sales for a FinTech company",
            "location": "Singapore",
            "chronic_condition": "Mild Hypertension",
            "goals": [
                "Reduce risk of heart disease",
                "Enhance cognitive function and focus",
                "Implement regular health screenings"
            ],
            "travel_pattern": "Frequent international travel to UK, US, South Korea, Jakarta"
        }

        # Journey structure will be initialized when needed
        self.weeks_structure = None
        self.journey_data = []

    async def _safe_llm_call(self, messages: List, context: str = "LLM call", max_retries: int = 3, week_number: int = None, temperature: float = 0.7):
        """
        Make a safe LLM call with automatic key rotation on rate limits.
        """
        for attempt in range(max_retries):
            try:
                # Get current key info - use week-specific key if week_number provided
                if week_number is not None:
                    current_key, current_key_nickname = self.key_manager.get_key_for_week(week_number)
                else:
                    current_key, current_key_nickname = self.key_manager.get_current_week_key()
                
                # Update LLM if key changed
                if current_key != getattr(self.llm, 'groq_api_key', None):
                    print(f"🔄 Updating LLM to use key: {current_key_nickname}")
                    self.llm = ChatGroq(
                        model="llama3-8b-8192", 
                        temperature=temperature,
                        groq_api_key=current_key
                    )
                
                # Make the call
                response = await self.llm.ainvoke(messages)
                
                # Record successful usage
                self.key_manager.record_api_usage(current_key, success=True)
                
                return response
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Record failed usage - use correct key based on week_number
                if week_number is not None:
                    current_key, _ = self.key_manager.get_key_for_week(week_number)
                else:
                    current_key, _ = self.key_manager.get_current_week_key()
                self.key_manager.record_api_usage(current_key, success=False, error_message=str(e))
                
                print(f"❌ {context} failed (attempt {attempt + 1}): {e}")
                
                # Check if it's a rate limit error
                if "rate limit" in error_message or "429" in error_message:
                    if attempt < max_retries - 1:
                        try:
                            # Try to rotate to next available key
                            new_key, new_key_nickname = self.key_manager.rotate_to_next_key(current_key)
                            print(f"🔄 Rotated to key: {new_key_nickname}")
                            
                            # Update LLM with new key
                            self.llm = ChatGroq(
                                model="llama3-8b-8192", 
                                temperature=0.7,
                                groq_api_key=new_key
                            )
                            
                            # Wait before retrying
                            await asyncio.sleep(2)
                            continue
                            
                        except Exception as rotation_error:
                            print(f"❌ Could not rotate key: {rotation_error}")
                            if attempt == max_retries - 1:
                                raise e
                    else:
                        raise e
                else:
                    # Non-rate-limit error
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed {context} after {max_retries} attempts")

    async def _plan_8_month_journey(self) -> List[WeekContext]:
        """Plan the 8-month journey with deterministic structure that follows the user's high-level plan."""
        weeks = []
        start_date = datetime.now()

        # 8 months -> ~35 weeks
        total_weeks = 35

        # Biomarker weeks fixed at ~3 months and ~6 months (week numbers)
        biomarker_weeks = {12, 24}

        # Physical exam block: weeks 2-4 (collection and scans)
        physical_exam_weeks = set(range(2, 5))

        # Results sharing block: weeks 5-8 (intermittent sharing and commitments)
        results_sharing_weeks = set(range(5, 9))

        # Travel: choose at least 1 week per 4-week block, but avoid week 1 and physical exam weeks
        travel_weeks = set()
        for block_start in range(1, total_weeks + 1, 4):
            # choose a travel week in that 4-week block, but prefer not to pick week 1 or exam weeks
            candidates = [w for w in range(block_start, min(block_start + 4, total_weeks + 1))
                          if w != 1 and w not in physical_exam_weeks]
            if not candidates:
                continue
            travel_week = random.choice(candidates)
            travel_weeks.add(travel_week)

        # Exercise updates: every 2 weeks starting week 2 (so every even week >=2)
        exercise_update_weeks = set(range(2, total_weeks + 1, 2))

        # Plan adjustments: allow occasional plan adjustment weeks (but lower probability in early structured blocks)
        for week_num in range(1, total_weeks + 1):
            week_start = start_date + timedelta(weeks=week_num - 1)
            week_end = week_start + timedelta(days=6)

            # Determine base week_type with priority:
            # ONBOARDING > PHYSICAL_EXAM > BIOMARKER > RESULTS_SHARING > CHECKIN/FORTNIGHTLY_CALL > NORMAL
            week_type = WeekType.NORMAL
            additional_context = ""
            travel_destination = None

            if week_num == 1:
                week_type = WeekType.ONBOARDING
                additional_context = "Onboarding: member shares medical history, priorities, current diets, medications, and initial expectations. Establish baseline goals and data-sharing consents."
            elif week_num in physical_exam_weeks:
                week_type = WeekType.PHYSICAL_EXAM
                additional_context = "Physical exam and sample collection: blood panels, vitals, ECG if indicated, and baseline imaging/screening as required."
            elif week_num in biomarker_weeks:
                # biomarker report takes precedence over other statuses
                week_type = WeekType.BIOMARKER
                additional_context = f"Biomarker testing week (month {(week_num // 4)+1}): full diagnostic panel to assess progress on BP, lipids, glucose, inflammation and relevant markers."
            elif week_num in results_sharing_weeks:
                week_type = WeekType.RESULTS_SHARING
                additional_context = "Results sharing: intermittent sharing of lab findings with the member; triage into 'major issues', 'needs followup', or 'all okay' and collect commitments for interventions."
            elif 9 <= week_num <= 20:
                # sustained check-in phase with fortnightly clinical calls
                if week_num % 2 == 0:
                    week_type = WeekType.FORTNIGHTLY_CALL
                    additional_context = "Fortnightly clinical review call with the medical team to review progress, remove blockers, and change interventions where needed."
                else:
                    week_type = WeekType.CHECKIN
                    additional_context = "Weekly concierge/wellness officer check-in to remove blockers, follow up on tasks, and encourage adherence."
            else:
                # weeks after 20: mix of check-ins and normal weeks, occasional plan adjustments
                if random.random() < 0.18:
                    week_type = WeekType.PLAN_ADJUSTMENT
                    additional_context = "Plan adjustment week: modify interventions based on member feedback, travel, or adherence issues."
                elif week_num % 2 == 0:
                    week_type = WeekType.CHECKIN
                    additional_context = "Regular check-in week."
                else:
                    week_type = WeekType.NORMAL
                    additional_context = "Regular monitoring week; member follows established routine."

            # If travel is scheduled this week and it's not onboarding/physical_exam (avoid those), mark travel.
            if week_num in travel_weeks and week_type not in (WeekType.ONBOARDING, WeekType.PHYSICAL_EXAM, WeekType.BIOMARKER):
                week_type = WeekType.TRAVEL
                destinations = ["London, UK", "New York, US", "Seoul, South Korea", "Jakarta, Indonesia"]
                travel_destination = random.choice(destinations)
                # enrich context with travel-related details while preserving earlier context flavor
                additional_context = (f"Business travel to {travel_destination}. " +
                                      (additional_context + " ") if additional_context else "") + \
                                     "Challenges: time zone changes, meal variability, and altered exercise/medication timing."

            # If it's an exercise update week (every 2 weeks), append that detail to context
            if week_num in exercise_update_weeks:
                # Don't replace the context; append a note so it remains coherent with mission-critical phases
                additional_context = (additional_context + " " if additional_context else "") + \
                                     "Fortnightly exercise update planned based on wearable data (adjust intensity, recovery)."

            # Randomize plan adherence modestly within 30-70%
            plan_adherence = random.uniform(0.3, 0.7)

            week_context = WeekContext(
                week_number=week_num,
                week_type=week_type,
                date_range=(week_start, week_end),
                additional_context=additional_context,
                travel_destination=travel_destination,
                biomarker_focus=("Full diagnostic panel" if week_num in biomarker_weeks else None),
                plan_adherence=plan_adherence
            )

            weeks.append(week_context)

        return weeks

    async def _generate_biomarker_report(self, week_context: WeekContext) -> str:
        """Generate a biomarker report using LLM with key management"""
        prompt = f"""
        Generate a concise biomarker test report for Rohan Patel (46-year-old male with mild hypertension).

        Context: {week_context.additional_context}
        Week: {week_context.week_number}

        Include:
        - 3-5 key biomarkers relevant to his condition and goals
        - Realistic values showing gradual improvement or areas needing attention
        - Brief interpretation of results
        - 1-2 actionable recommendations

        Keep it concise (3-4 lines) and professional.
        """

        messages = [
            SystemMessage(content="You are a medical professional generating biomarker reports."),
            HumanMessage(content=prompt)
        ]

        response = await self._safe_llm_call(messages, f"Biomarker report for week {week_context.week_number}", week_number=week_context.week_number)
        return response.content.strip()

    def _extract_json_from_response(self, response_text: str) -> Dict[str, str]:
        """Extract JSON from LLM response using JSON prompting."""
        import re
        
        # Try to find JSON in the response
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                parsed_json = json.loads(match)
                # Validate that it has the expected structure (q1, q2, etc.)
                if all(key.startswith('q') and key[1:].isdigit() for key in parsed_json.keys()):
                    return parsed_json
            except json.JSONDecodeError:
                continue
        
        # Fallback: try to extract from code block
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        
        for match in code_matches:
            try:
                parsed_json = json.loads(match)
                if all(key.startswith('q') and key[1:].isdigit() for key in parsed_json.keys()):
                    return parsed_json
            except json.JSONDecodeError:
                continue
        
        # Final fallback: return empty dict if no valid JSON found
        print(f"⚠️ Could not extract valid JSON from response: {response_text[:200]}...")
        return {}

    async def _generate_weekly_queries(self, week_context: WeekContext) -> List[str]:
        """Generate 5 realistic patient queries for a specific week with key management.

        Produces queries that sound like the patient (Rohan) reporting feelings/symptoms,
        asking data-interpretation questions, logistical scheduling items, and casual
        or urgent notes.
        """

        # pick a temperature each call so outputs vary
        temperature = 0.8

        base_prompt = f"""
        You are ROLEPLAYING as Rohan Patel (46yo, male, Regional Head of Sales, analytical,
        frequent traveler, mild hypertension).
        Keep messages brief (one or two sentences each), realistic, and specific to Rohan's
        profile.
        Week Context: {week_context.additional_context}
        Week Type: {week_context.week_type.value}
        Plan Adherence: {week_context.plan_adherence * 100:.0f}%

        Patient Snapshot (for voice and specificity):
        - Preferred name: Rohan Patel (reply in first-person where relevant, e.g. "I'm feeling...")
        - Lives in Singapore, frequent travel hubs: UK, US, South Korea, Jakarta
        - Goals: reduce heart disease risk, improve cognition
        - Wearable: Garmin (sleep, HR, HRV), considering Oura
        - Personality: analytical, likes data and concise recommendations
        - Typical availability: mornings for quick check-ins; PA (Sarah) handles scheduling

        IMPORTANT: Tailor queries to the specified Week Type.
        """

        # add week-specific behavioral/symptom hooks
        if week_context.week_type == WeekType.ONBOARDING:
            specific_prompt = base_prompt + f"""
            This is an ONBOARDING week. Include questions about:
            - How baseline measures will be used,
            - Permission/data-sharing wording (concise),
            - What initial tests mean for immediate next steps.
            """
        elif week_context.week_type == WeekType.PHYSICAL_EXAM:
            specific_prompt = base_prompt + f"""
            This is a PHYSICAL EXAM week. Include questions about:
            - Preparing for fasting tests, medication restrictions,
            - Concerns like "what if I can't fast because of travel" or "should I stop BP meds?",
            - Quick symptom check-ins (e.g., palpitations before/after tests).
            """
        elif week_context.week_type == WeekType.TRAVEL:
            dest = getattr(week_context, "travel_destination", None) or "current destination"
            specific_prompt = base_prompt + f"""
            This is a TRAVEL week to {dest}. Include travel-specific items such as:
            - Medication timing across time zones,
            - Sleep/jetlag effects on BP or concentration,
            - Wearable anomalies in flight, and quick in-room workouts.
            """
        else:
            specific_prompt = base_prompt + f"""
            General week: include a mix of:
            - Symptom reports (restless, palpitations, headache, tiredness),
            - Quick data interpretation asks (changes in HR/HRV, steps/VO2 trends),
            - Nutrition or supplement timing questions,
            - One question about scheduling or urgent escalation if needed.
            """

        specific_prompt += """

        RESPONSE FORMAT (strict JSON): Return exactly 5 items with keys q1..q5.
        {
          "q1": "first patient-style message",
          "q2": "second patient-style message",
          "q3": "third patient-style message",
          "q4": "fourth patient-style message",
          "q5": "fifth patient-style message"
        }

        - Each value should be 8-120 characters, natural first-person style where appropriate.
        - Include a variety: symptom report, data question, logistics, travel, and short ask for action.
        """

        messages = [
            SystemMessage(content=(
                "You are a patient-voice simulator. Produce short, realistic messages "
                "as if sent by the patient Rohan Patel. Always respond in valid JSON "
                "following the required schema."
            )),
            HumanMessage(content=specific_prompt)
        ]

        # pass temperature through to the LLM call so outputs vary
        # (assumes _safe_llm_call accepts a temperature kwarg)
        response = await self._safe_llm_call(
            messages,
            f"Weekly queries for week {week_context.week_number}",
            week_number=week_context.week_number,
            temperature=temperature
        )

        # Extract JSON from the response
        queries_json = self._extract_json_from_response(response.content)

        # Normalize into ordered list q1..q5; robust to missing keys / different keys
        queries: List[str] = []
        if queries_json and isinstance(queries_json, dict):
            # pick keys that match q<number>
            q_items = []
            for k, v in queries_json.items():
                if isinstance(k, str) and k.lower().startswith('q'):
                    suffix = k[1:]
                    try:
                        idx = int(suffix)
                    except Exception:
                        idx = 999
                    q_items.append((idx, str(v).strip()))
            # sort and take first 5 unique non-empty
            q_items = sorted(q_items, key=lambda x: x[0])
            seen = set()
            for _, text in q_items:
                if text and text not in seen:
                    queries.append(text)
                    seen.add(text)
                if len(queries) >= 5:
                    break

        # Fallback to text parsing or padding if extraction failed or <5 items returned
        # if len(queries) < 5:
        #     # try old text-parsing fallback first
        #     print(f"⚠️ JSON extraction incomplete for week {week_context.week_number}, attempting text fallback/pad")
        #     for line in response.content.strip().split('\n'):
        #         if len(queries) >= 5:
        #             break
        #         line = line.strip()
        #         if not line:
        #             continue
        #         # Accept both "1. q" and "- q" styles
        #         if line[0].isdigit() and '.' in line:
        #             candidate = line.split('.', 1)[1].strip()
        #         elif line.startswith('-'):
        #             candidate = line[1:].strip()
        #         else:
        #             # fallback: treat the whole line as a candidate (if short)
        #             candidate = line
        #         if candidate and candidate not in queries:
        #             queries.append(candidate)

        # Ensure exactly 5 items and return
        return queries

    async def _run_orchestrator_conversation(self, query: str, week_context: WeekContext) -> Dict[str, Any]:
        """Run a single conversation through the enhanced orchestrator"""
        try:
            conversation_result = await self.orchestrator.start_conversation(query)

            # Add week context to the conversation
            conversation_result["week_info"] = {
                "week_number": week_context.week_number,
                "week_type": week_context.week_type.value,
                "date_range": [
                    week_context.date_range[0].isoformat(),
                    week_context.date_range[1].isoformat()
                ],
                "context": week_context.additional_context,
                "travel_destination": week_context.travel_destination,
                "plan_adherence": week_context.plan_adherence
            }

            # Add journey metadata for better tracking
            conversation_result["journey_metadata"] = {
                "patient_name": self.patient_profile["name"],
                "chronic_condition": self.patient_profile["chronic_condition"],
                "conversation_in_journey": True,
                "synthesis_timestamp": datetime.now().isoformat()
            }

            return conversation_result

        except Exception as e:
            print(f"Error in orchestrator conversation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "week_info": {
                    "week_number": week_context.week_number,
                    "week_type": week_context.week_type.value
                },
                "journey_metadata": {
                    "patient_name": self.patient_profile["name"],
                    "synthesis_timestamp": datetime.now().isoformat()
                }
            }

    async def synthesize_week(self, week_context: WeekContext) -> Dict[str, Any]:
        """Synthesize all conversations and data for a single week"""
        print(f"\n🗓️ Synthesizing Week {week_context.week_number} ({week_context.week_type.value})")
        print(f"📅 {week_context.date_range[0].strftime('%Y-%m-%d')} to {week_context.date_range[1].strftime('%Y-%m-%d')}")
        print(f"📝 Context: {week_context.additional_context}")

        # Get the key for this specific journey week (not current calendar week)
        current_key, current_key_nickname = self.key_manager.get_key_for_week(week_context.week_number)
        print(f"🔑 Using API key for week {week_context.week_number}: {current_key_nickname}")
        
        # Update LLM with the correct key for this week
        self.llm = ChatGroq(
            model="llama3-8b-8192", 
            temperature=0.7,
            groq_api_key=current_key
        )

        # Start new week in journey logger
        self.journey_logger.start_new_week(
            week_number=week_context.week_number,
            week_type=week_context.week_type.value,
            week_context=week_context.additional_context,
            date_range=week_context.date_range,
            travel_destination=week_context.travel_destination,
            plan_adherence=week_context.plan_adherence
        )

        week_data = {
            "week_number": week_context.week_number,
            "week_type": week_context.week_type.value,
            "date_range": [
                week_context.date_range[0].isoformat(),
                week_context.date_range[1].isoformat()
            ],
            "context": week_context.additional_context,
            "travel_destination": week_context.travel_destination,
            "plan_adherence": week_context.plan_adherence,
            "conversations": [],
            "biomarker_report": None,
            "api_key_used": current_key_nickname
        }

        # Generate biomarker report if it's a biomarker week
        if week_context.week_type == WeekType.BIOMARKER:
            print("🔬 Generating biomarker report...")
            biomarker_report = await self._generate_biomarker_report(week_context)
            week_data["biomarker_report"] = biomarker_report
            # Add biomarker report to journey logger
            self.journey_logger.add_biomarker_report(biomarker_report, week_context.week_number)
            print(f"📊 Biomarker Report: {biomarker_report[:100]}...")

        # Generate 5 queries for the week
        print("💭 Generating weekly queries...")
        queries = await self._generate_weekly_queries(week_context)

        # Run each query through the orchestrator
        for i, query in enumerate(queries, 1):
            print(f"  🔄 Processing query {i}/5: {query[:60]}...")
            conversation = await self._run_orchestrator_conversation(query, week_context)
            conversation["query_number"] = i
            week_data["conversations"].append(conversation)

            # Add conversation to journey logger after each orchestrator execution
            self.journey_logger.add_conversation(conversation, week_context.week_number)

            # Small delay between conversations to be respectful to API
            await asyncio.sleep(1)

        # Complete the week in journey logger
        week_summary = f"Week {week_context.week_number} completed with {len(queries)} conversations. Type: {week_context.week_type.value}. Key used: {current_key_nickname}"
        self.journey_logger.complete_week(week_context.week_number, week_summary)

        return week_data

    async def synthesize_full_journey(self) -> Dict[str, Any]:
        """Synthesize the complete 8-month patient journey with enhanced key management"""
        print("🚀 Starting Enhanced 8-Month Patient Journey Synthesis with Load Balancing")
        print("=" * 70)

        # Print initial key status
        self.key_manager.print_usage_summary()

        journey_start_time = datetime.now()

        # Initialize journey structure if not already done
        if self.weeks_structure is None:
            print("📋 Planning 8-month journey structure...")
            self.weeks_structure = await self._plan_8_month_journey()

        full_journey_data = {
            "patient_profile": self.patient_profile,
            "journey_metadata": {
                "total_weeks": len(self.weeks_structure),
                "synthesis_start_time": journey_start_time.isoformat(),
                "key_management_enabled": True,
                "constraints": {
                    "biomarker_tests": "Every 3 months",
                    "conversations_per_week": 5,
                    "hours_committed_per_week": 5,
                    "exercise_updates": "Every 2 weeks",
                    "travel_frequency": "1 week out of 4",
                    "plan_adherence": "~50%",
                    "chronic_condition": self.patient_profile["chronic_condition"]
                }
            },
            "weeks": []
        }

        # Synthesize each week
        for week_context in self.weeks_structure:
            try:
                week_data = await self.synthesize_week(week_context)
                full_journey_data["weeks"].append(week_data)

                # Print progress every 5 weeks with key usage
                if week_context.week_number % 5 == 0:
                    stats = self.journey_logger.get_journey_stats()
                    key_summary = self.key_manager.get_usage_summary()
                    
                    print(f"📊 Progress Update - Week {week_context.week_number}:")
                    print(f"   💬 Total conversations: {stats['total_conversations']}")
                    print(f"   ✅ Completed weeks: {stats['completed_weeks']}")
                    print(f"   📈 Avg conversations/week: {stats['average_conversations_per_week']:.1f}")
                    print(f"   🔑 Active keys: {key_summary['active_keys']}/{key_summary['total_keys']}")
                    print(f"   🚫 Rate limited keys: {key_summary['rate_limited_keys']}")

                # Save intermediate progress every 5 weeks
                if week_context.week_number % 5 == 0:
                    await self._save_intermediate_progress(full_journey_data, week_context.week_number)

                # Check and reset rate limits
                self.key_manager.check_and_reset_rate_limits()

            except Exception as e:
                print(f"❌ Error synthesizing week {week_context.week_number}: {e}")
                # Continue with next week
                continue

        # Finalize journey data
        journey_end_time = datetime.now()
        full_journey_data["journey_metadata"]["synthesis_end_time"] = journey_end_time.isoformat()
        full_journey_data["journey_metadata"]["total_synthesis_time"] = str(journey_end_time - journey_start_time)

        # Get final statistics
        final_stats = self.journey_logger.get_journey_stats()
        final_key_stats = self.key_manager.get_usage_summary()

        # Calculate summary statistics
        total_conversations = final_stats["total_conversations"]
        biomarker_weeks = final_stats["week_type_distribution"].get("biomarker", 0)
        travel_weeks = final_stats["week_type_distribution"].get("travel", 0)

        full_journey_data["journey_metadata"]["summary_stats"] = {
            "total_conversations": total_conversations,
            "biomarker_test_weeks": biomarker_weeks,
            "travel_weeks": travel_weeks,
            "average_conversations_per_week": final_stats["average_conversations_per_week"],
            "week_type_distribution": final_stats["week_type_distribution"],
            "api_key_usage": final_key_stats
        }

        # Save complete journey
        await self._save_complete_journey(full_journey_data)

        # Export journey summary
        summary_path = self.journey_logger.export_journey_summary()

        print("=" * 70)
        print("🎉 Enhanced 8-Month Patient Journey Synthesis Complete!")
        print(f"📊 Total weeks: {final_stats['total_weeks']}")
        print(f"💬 Total conversations: {total_conversations}")
        print(f"🔬 Biomarker test weeks: {biomarker_weeks}")
        print(f"✈️ Travel weeks: {travel_weeks}")
        print(f"🔑 API keys used: {final_key_stats['total_keys']}")
        print(f"📄 Journey summary: {summary_path}")
        print(f"🗃️ Complete journey log: {self.journey_logger.journey_file_path}")

        # Print final key usage summary
        print("\n📊 Final API Key Usage Summary:")
        self.key_manager.print_usage_summary()

        return full_journey_data

    async def _save_intermediate_progress(self, journey_data: Dict[str, Any], week_number: int):
        """Save intermediate progress with key usage info"""
        key_summary = self.key_manager.get_usage_summary()
        
        filename = f"journey_progress_week_{week_number}_with_keys.json"
        filepath = os.path.join(self.output_dir, filename)

        progress_data = journey_data.copy()
        progress_data["key_usage_at_checkpoint"] = key_summary

        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)

        print(f"💾 Intermediate progress saved: {filename}")

    async def _save_complete_journey(self, journey_data: Dict[str, Any]):
        """Save the complete journey data with key management info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_8_month_journey_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(journey_data, f, indent=2, default=str)

        print(f"💾 Complete enhanced journey saved: {filename}")
        return filepath

    def generate_summary_report(self, journey_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary report with key usage info"""
        report = []
        report.append("# Enhanced 8-Month Patient Journey Summary Report")
        report.append("=" * 60)

        # Patient info
        patient = journey_data["patient_profile"]
        report.append(f"\n## Patient: {patient['name']}")
        report.append(f"- Age: {patient['age']}")
        report.append(f"- Occupation: {patient['occupation']}")
        report.append(f"- Location: {patient['location']}")
        report.append(f"- Chronic Condition: {patient['chronic_condition']}")

        # Journey stats
        stats = journey_data["journey_metadata"]["summary_stats"]
        report.append(f"\n## Journey Statistics")
        report.append(f"- Total Weeks: {len(journey_data['weeks'])}")
        report.append(f"- Total Conversations: {stats['total_conversations']}")
        report.append(f"- Average Conversations per Week: {stats['average_conversations_per_week']:.1f}")
        report.append(f"- Biomarker Test Weeks: {stats['biomarker_test_weeks']}")
        report.append(f"- Travel Weeks: {stats['travel_weeks']}")

        # API Key usage stats
        if 'api_key_usage' in stats:
            key_stats = stats['api_key_usage']
            report.append(f"\n## API Key Usage Statistics")
            report.append(f"- Total Keys Used: {key_stats['total_keys']}")
            report.append(f"- Active Keys: {key_stats['active_keys']}")
            report.append(f"- Rate Limited Keys: {key_stats['rate_limited_keys']}")
            
            report.append(f"\n### Individual Key Performance:")
            for key_detail in key_stats.get('key_details', []):
                status_emoji = {
                    'active': '✅',
                    'rate_limited': '🚫', 
                    'error': '❌',
                    'disabled': '⏸️'
                }.get(key_detail['status'], '❓')
                
                report.append(f"{status_emoji} {key_detail['nickname']}: {key_detail['total_requests']} requests, {key_detail['error_count']} errors")

        # Week-by-week breakdown
        report.append(f"\n## Week-by-Week Breakdown")
        for week in journey_data["weeks"]:
            report.append(f"\n### Week {week['week_number']} ({week['week_type']})")
            report.append(f"Context: {week['context']}")
            if week.get("travel_destination"):
                report.append(f"Travel Destination: {week['travel_destination']}")
            report.append(f"Conversations: {len(week.get('conversations', []))}")
            if week.get("api_key_used"):
                report.append(f"API Key Used: {week['api_key_used']}")
            if week.get("biomarker_report"):
                report.append(f"Biomarker Report: {week['biomarker_report'][:100]}...")

        return "\n".join(report)


# Utility function to run the enhanced synthesizer
async def run_enhanced_8_month_synthesis(output_dir: str = "enhanced_synthesized_journey") -> str:
    """Run the complete enhanced 8-month patient journey synthesis with key management"""
    synthesizer = EnhancedPatientJourneySynthesizer(output_dir)
    journey_data = await synthesizer.synthesize_full_journey()

    # Generate and save summary report
    summary = synthesizer.generate_summary_report(journey_data)
    summary_path = os.path.join(output_dir, "enhanced_journey_summary_report.md")
    with open(summary_path, 'w') as f:
        f.write(summary)

    return summary_path


if __name__ == "__main__":
    asyncio.run(run_enhanced_8_month_synthesis())
