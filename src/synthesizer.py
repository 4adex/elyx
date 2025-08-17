"""
8-Month Patient Journey Synthesizer for Elyx

This module simulates an 8-month patient journey with realistic constraints:
- Biomarker tests every 3 months (randomly placed within week ranges)
- 5 conversations per week on average initiated by member
- Member commits 5 hours/week to follow the plan
- Exercise updates every 2 weeks
- Travel 1 week out of every 4 weeks
- Member follows plan ~50% of the time properly
- Member has one chronic condition to manage
"""

import asyncio
import random
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .orchestation import OrchestatedAgent
from .conversation_logger import ConversationLogger
from .journey_logger import JourneyLogger


class WeekType(Enum):
    NORMAL = "normal"
    TRAVEL = "travel"
    BIOMARKER = "biomarker"
    EXERCISE_UPDATE = "exercise_update"
    PLAN_ADJUSTMENT = "plan_adjustment"


@dataclass
class WeekContext:
    week_number: int
    week_type: WeekType
    date_range: Tuple[datetime, datetime]
    additional_context: str
    travel_destination: str = None
    biomarker_focus: str = None
    plan_adherence: float = 0.5  # 50% default adherence


class PatientJourneySynthesizer:
    def __init__(self, output_dir: str = "synthesized_journey"):
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
        self.orchestrator = OrchestatedAgent()
        self.conversation_logger = ConversationLogger(output_dir)
        self.journey_logger = JourneyLogger(output_dir)  # New journey logger
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

    async def _plan_8_month_journey(self) -> List[WeekContext]:
        """Plan the 8-month journey with all constraints"""
        weeks = []
        start_date = datetime.now()
        
        # Calculate 8 months = ~35 weeks
        total_weeks = 35
        
        # Mark biomarker weeks (every 12-13 weeks, with some randomness)
        biomarker_weeks = set()
        for month in [3, 6]:  # 3 months and 6 months
            week_num = month * 4 + random.randint(-1, 1)  # Add some randomness
            if 1 <= week_num <= total_weeks:
                biomarker_weeks.add(week_num)
        
        # Mark travel weeks (1 out of every 4 weeks)
        travel_weeks = set()
        for cycle_start in range(1, total_weeks + 1, 4):
            travel_week = cycle_start + random.randint(0, 3)
            if travel_week <= total_weeks:
                travel_weeks.add(travel_week)
        
        # Mark exercise update weeks (every 2 weeks)
        exercise_update_weeks = set(range(2, total_weeks + 1, 2))
        
        # Generate weeks
        for week_num in range(1, total_weeks + 1):
            week_start = start_date + timedelta(weeks=week_num - 1)
            week_end = week_start + timedelta(days=6)
            
            # Determine week type and context
            if week_num in biomarker_weeks:
                week_type = WeekType.BIOMARKER
                context = await self._generate_biomarker_context(week_num)
                destination = None
            elif week_num in travel_weeks:
                week_type = WeekType.TRAVEL
                context, destination = await self._generate_travel_context(week_num)
            elif week_num in exercise_update_weeks:
                week_type = WeekType.EXERCISE_UPDATE
                context = await self._generate_exercise_update_context(week_num)
                destination = None
            elif random.random() < 0.3:  # 30% chance of plan adjustment
                week_type = WeekType.PLAN_ADJUSTMENT
                context = await self._generate_plan_adjustment_context(week_num)
                destination = None
            else:
                week_type = WeekType.NORMAL
                context = await self._generate_normal_context(week_num)
                destination = None
            
            week_context = WeekContext(
                week_number=week_num,
                week_type=week_type,
                date_range=(week_start, week_end),
                additional_context=context,
                travel_destination=destination,
                plan_adherence=random.uniform(0.3, 0.7)  # 30-70% adherence variation
            )
            
            weeks.append(week_context)
        
        return weeks

    async def _generate_biomarker_context(self, week_num: int) -> str:
        """Generate biomarker context using LLM"""
        prompt = f"""
        Generate a realistic biomarker testing context for week {week_num} of Rohan Patel's health journey.
        
        Patient Profile:
        - 46-year-old male with mild hypertension
        - FinTech executive with high-stress job
        - Goal: reduce heart disease risk, improve cognitive function
        
        Create a 1-2 sentence context that includes:
        - What specific biomarkers or health areas will be tested
        - Why this timing makes sense in his health journey
        - What progress or concerns might be monitored
        
        Keep it realistic and specific to his condition and goals.
        Format: "Week {week_num}: [context]"
        """
        
        messages = [
            SystemMessage(content="You are a health coach planning biomarker testing for a patient's health journey."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def _generate_travel_context(self, week_num: int) -> Tuple[str, str]:
        """Generate travel context using LLM"""
        destinations = ["London, UK", "New York, US", "Seoul, South Korea", "Jakarta, Indonesia"]
        destination = random.choice(destinations)
        
        prompt = f"""
        Generate a realistic business travel context for week {week_num} of Rohan Patel's health journey.
        
        Patient Profile:
        - 46-year-old FinTech executive from Singapore
        - Managing mild hypertension
        - Travels frequently for work
        - Committed to maintaining health routine while traveling
        
        Travel Details:
        - Destination: {destination}
        - Business travel (meetings, conferences, client visits)
        
        Create a 1-2 sentence context that includes:
        - Specific health challenges this travel might present
        - How it affects his routine (exercise, meals, medication timing)
        - Any particular concerns for someone with hypertension
        
        Keep it realistic and specific to business travel challenges.
        Format: "Week {week_num}: Business travel to {destination}. [specific challenge context]"
        """
        
        messages = [
            SystemMessage(content="You are a health context generator for business travel challenges."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip(), destination

    async def _generate_exercise_update_context(self, week_num: int) -> str:
        """Generate exercise update context using LLM"""
        prompt = f"""
        Generate a realistic exercise program update context for week {week_num} of Rohan Patel's health journey.
        
        Patient Profile:
        - 46-year-old FinTech executive with mild hypertension
        - Goal: reduce heart disease risk, improve cognitive function
        - Uses Garmin for fitness tracking
        - Busy schedule requiring efficient workouts
        - Values data-driven approach to fitness
        
        Create a 1-2 sentence context that includes:
        - What aspect of his exercise routine is being updated/progressed
        - Why this timing makes sense (every 2 weeks progression)
        - How it relates to his health goals and current fitness level
        - Reference to data/metrics if relevant
        
        Keep it realistic for a busy executive's workout routine.
        Format: "Week {week_num}: Exercise program update. [specific update context]"
        """
        
        messages = [
            SystemMessage(content="You are a fitness coach planning exercise progressions for a busy executive."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def _generate_plan_adjustment_context(self, week_num: int) -> str:
        """Generate plan adjustment context using LLM"""
        prompt = f"""
        Generate a realistic health plan adjustment context for week {week_num} of Rohan Patel's health journey.
        
        Patient Profile:
        - 46-year-old FinTech executive with mild hypertension
        - High-stress job with varying schedule demands
        - Managing health routine alongside work commitments
        - Goal: reduce heart disease risk, improve cognitive function
        
        Create a 1-2 sentence context that includes:
        - A realistic reason why his health plan needs adjustment
        - How this relates to his work/life balance challenges
        - What specific aspects of his routine might need modification
        - Maintain focus on his health goals despite challenges
        
        Consider factors like:
        - Work deadlines and stress
        - Family commitments
        - Seasonal changes
        - Minor setbacks or obstacles
        - Schedule conflicts
        
        Format: "Week {week_num}: Plan adjustment needed [reason and context]"
        """
        
        messages = [
            SystemMessage(content="You are a health coach helping adapt health plans to real-life challenges."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def _generate_normal_context(self, week_num: int) -> str:
        """Generate normal week context using LLM"""
        prompt = f"""
        Generate a realistic normal week context for week {week_num} of Rohan Patel's health journey.
        
        Patient Profile:
        - 46-year-old FinTech executive with mild hypertension
        - Following established health routine
        - Goal: reduce heart disease risk, improve cognitive function
        - Uses wearable devices for health tracking
        - Values consistency and progress monitoring
        
        Create a 1-2 sentence context that includes:
        - Focus on routine maintenance and consistency
        - Reference to ongoing health activities (monitoring, routine adherence)
        - Positive reinforcement of established habits
        - How this week supports his long-term health goals
        
        Keep it realistic for a normal week without special events.
        Consider factors like:
        - Routine health monitoring
        - Steady progress maintenance
        - Regular engagement with health team
        - Daily metric tracking
        - Motivation and support needs
        
        Format: "Week {week_num}: [normal week context]"
        """
        
        messages = [
            SystemMessage(content="You are a health coach providing context for routine health maintenance weeks."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def _generate_biomarker_report(self, week_context: WeekContext) -> str:
        """Generate a biomarker report using LLM"""
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
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    async def _generate_weekly_queries(self, week_context: WeekContext) -> List[str]:
        """Generate 5 realistic patient queries for a specific week"""
        
        base_prompt = f"""
        Generate 5 realistic health-related queries that Rohan Patel (46-year-old FinTech executive with mild hypertension) 
        might ask the Elyx team during this week.
        
        Week Context: {week_context.additional_context}
        Week Type: {week_context.week_type.value}
        Plan Adherence: {week_context.plan_adherence * 100:.0f}%
        
        Patient Profile:
        - Lives in Singapore, travels frequently
        - Managing mild hypertension
        - Goal: reduce heart disease risk, improve cognitive function
        - Uses Garmin for fitness tracking
        - Analytical personality, prefers data-driven approaches
        
        """
        
        if week_context.week_type == WeekType.TRAVEL:
            specific_prompt = f"""
            {base_prompt}
            
            This is a TRAVEL week to {week_context.travel_destination}.
            Include travel-specific queries about:
            - Maintaining routine while traveling
            - Food/exercise options at destination
            - Medication timing with timezone changes
            - Managing stress during business trips
            - Wearable data interpretation during travel
            """
        elif week_context.week_type == WeekType.BIOMARKER:
            specific_prompt = f"""
            {base_prompt}
            
            This is a BIOMARKER TESTING week.
            Include queries about:
            - Preparing for blood tests
            - Understanding previous results
            - Questions about specific biomarkers
            - Concerns about trends in health metrics
            - Follow-up actions based on results
            """
        elif week_context.week_type == WeekType.EXERCISE_UPDATE:
            specific_prompt = f"""
            {base_prompt}
            
            This is an EXERCISE UPDATE week.
            Include queries about:
            - Progress on current workout routine
            - Form and technique questions
            - Recovery and rest day protocols
            - Adjusting intensity based on performance
            - Integration with wearable data
            """
        else:
            specific_prompt = f"""
            {base_prompt}
            
            Include general queries about:
            - Daily health routine questions
            - Nutrition and meal planning
            - Supplement timing and effectiveness
            - Sleep optimization
            - Managing work stress and health balance
            """
        
        specific_prompt += """
        
        Return exactly 5 queries, each on a new line, numbered 1-5.
        Make queries conversational and specific to Rohan's situation.
        Mix simple questions with more complex health optimization queries.
        """
        
        messages = [
            SystemMessage(content="You are helping generate realistic patient queries for a health journey simulation."),
            HumanMessage(content=specific_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse the response to extract queries
        queries = []
        for line in response.content.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                query = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                queries.append(query)
        
        # Ensure we have exactly 5 queries
        if len(queries) < 5:
            # Add generic queries if needed
            generic_queries = [
                "How are my blood pressure trends looking this week?",
                "Should I adjust my medication timing?",
                "Any recommendations for stress management?",
                "How's my sleep quality affecting my health goals?",
                "What should I focus on this week for better results?"
            ]
            while len(queries) < 5:
                queries.append(random.choice(generic_queries))
        
        return queries[:5]

    async def _run_orchestrator_conversation(self, query: str, week_context: WeekContext) -> Dict[str, Any]:
        """Run a single conversation through the orchestrator"""
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
            "biomarker_report": None
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
        
        # Complete the week in journey logger
        week_summary = f"Week {week_context.week_number} completed with {len(queries)} conversations. Type: {week_context.week_type.value}. Context: {week_context.additional_context[:100]}..."
        self.journey_logger.complete_week(week_context.week_number, week_summary)
        
        return week_data

    async def synthesize_full_journey(self) -> Dict[str, Any]:
        """Synthesize the complete 8-month patient journey"""
        print("🚀 Starting 8-Month Patient Journey Synthesis")
        print("=" * 60)
        
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
                
                # Print progress every 5 weeks
                if week_context.week_number % 5 == 0:
                    stats = self.journey_logger.get_journey_stats()
                    print(f"📊 Progress Update - Week {week_context.week_number}:")
                    print(f"   💬 Total conversations: {stats['total_conversations']}")
                    print(f"   ✅ Completed weeks: {stats['completed_weeks']}")
                    print(f"   📈 Avg conversations/week: {stats['average_conversations_per_week']:.1f}")
                
                # Save intermediate progress every 5 weeks
                if week_context.week_number % 5 == 0:
                    await self._save_intermediate_progress(full_journey_data, week_context.week_number)
                
            except Exception as e:
                print(f"❌ Error synthesizing week {week_context.week_number}: {e}")
                # Continue with next week
                continue
        
        # Finalize journey data
        journey_end_time = datetime.now()
        full_journey_data["journey_metadata"]["synthesis_end_time"] = journey_end_time.isoformat()
        full_journey_data["journey_metadata"]["total_synthesis_time"] = str(journey_end_time - journey_start_time)
        
        # Get final statistics from journey logger
        final_stats = self.journey_logger.get_journey_stats()
        
        # Calculate summary statistics
        total_conversations = final_stats["total_conversations"]
        biomarker_weeks = final_stats["week_type_distribution"].get("biomarker", 0)
        travel_weeks = final_stats["week_type_distribution"].get("travel", 0)
        
        full_journey_data["journey_metadata"]["summary_stats"] = {
            "total_conversations": total_conversations,
            "biomarker_test_weeks": biomarker_weeks,
            "travel_weeks": travel_weeks,
            "average_conversations_per_week": final_stats["average_conversations_per_week"],
            "week_type_distribution": final_stats["week_type_distribution"]
        }
        
        # Save complete journey
        await self._save_complete_journey(full_journey_data)
        
        # Export journey summary
        summary_path = self.journey_logger.export_journey_summary()
        
        print("=" * 60)
        print("🎉 8-Month Patient Journey Synthesis Complete!")
        print(f"📊 Total weeks: {final_stats['total_weeks']}")
        print(f"💬 Total conversations: {total_conversations}")
        print(f"🔬 Biomarker test weeks: {biomarker_weeks}")
        print(f"✈️ Travel weeks: {travel_weeks}")
        print(f"📄 Journey summary: {summary_path}")
        print(f"🗃️ Complete journey log: {self.journey_logger.journey_file_path}")
        
        return full_journey_data

    async def _save_intermediate_progress(self, journey_data: Dict[str, Any], week_number: int):
        """Save intermediate progress"""
        filename = f"journey_progress_week_{week_number}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(journey_data, f, indent=2, default=str)
        
        print(f"💾 Intermediate progress saved: {filename}")

    async def _save_complete_journey(self, journey_data: Dict[str, Any]):
        """Save the complete journey data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_8_month_journey_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(journey_data, f, indent=2, default=str)
        
        print(f"💾 Complete journey saved: {filename}")
        return filepath

    def generate_summary_report(self, journey_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append("# 8-Month Patient Journey Summary Report")
        report.append("=" * 50)
        
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
        
        # Week-by-week breakdown
        report.append(f"\n## Week-by-Week Breakdown")
        for week in journey_data["weeks"]:
            report.append(f"\n### Week {week['week_number']} ({week['week_type']})")
            report.append(f"Context: {week['context']}")
            if week.get("travel_destination"):
                report.append(f"Travel Destination: {week['travel_destination']}")
            report.append(f"Conversations: {len(week.get('conversations', []))}")
            if week.get("biomarker_report"):
                report.append(f"Biomarker Report: {week['biomarker_report'][:100]}...")
        
        return "\n".join(report)


# Utility function to run the synthesizer
async def run_8_month_synthesis(output_dir: str = "synthesized_journey") -> str:
    """Run the complete 8-month patient journey synthesis"""
    synthesizer = PatientJourneySynthesizer(output_dir)
    journey_data = await synthesizer.synthesize_full_journey()
    
    # Generate and save summary report
    summary = synthesizer.generate_summary_report(journey_data)
    summary_path = os.path.join(output_dir, "journey_summary_report.md")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return summary_path


if __name__ == "__main__":
    asyncio.run(run_8_month_synthesis())