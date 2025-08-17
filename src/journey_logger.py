"""
8-Month Journey Logger for Elyx

This module handles logging of all conversations in an 8-month patient journey
to a single JSON file with weekly separators and chronological organization.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
import copy


class JourneyLogger:
    """Handles logging of 8-month patient journey conversations to a single JSON file"""
    
    def __init__(self, log_dir: str = "logs", journey_filename: str = None):
        """Initialize the journey logger
        
        Args:
            log_dir: Directory for log files
            journey_filename: Custom filename for the 8-month journey log
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped filename if not provided
        if journey_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            journey_filename = f"8_month_journey_{timestamp}.json"
        
        self.journey_file_path = os.path.join(log_dir, journey_filename)
        self.lock = threading.Lock()  # Thread safety for concurrent writes
        
        # Initialize the journey file if it doesn't exist
        self._initialize_journey_file()
    
    def _initialize_journey_file(self):
        """Initialize the 8-month journey JSON file with base structure"""
        if not os.path.exists(self.journey_file_path):
            initial_data = {
                "journey_metadata": {
                    "patient_profile": {
                        "name": "Rohan Patel",
                        "age": 46,
                        "occupation": "Regional Head of Sales for a FinTech company",
                        "location": "Singapore",
                        "chronic_condition": "Mild Hypertension",
                        "start_date": datetime.now().isoformat(),
                        "goals": [
                            "Reduce risk of heart disease",
                            "Enhance cognitive function and focus",
                            "Implement regular health screenings"
                        ]
                    },
                    "journey_constraints": {
                        "total_duration_months": 8,
                        "conversations_per_week": 5,
                        "biomarker_tests": "Every 3 months",
                        "exercise_updates": "Every 2 weeks",
                        "travel_frequency": "1 week out of 4",
                        "plan_adherence_target": "~50%"
                    },
                    "file_created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_conversations": 0,
                    "total_weeks": 0
                },
                "weekly_conversations": []
            }
            
            with open(self.journey_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    def _load_journey_data(self) -> Dict[str, Any]:
        """Load the current journey data from file"""
        try:
            with open(self.journey_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading journey data: {e}")
            self._initialize_journey_file()
            return self._load_journey_data()
    
    def _save_journey_data(self, data: Dict[str, Any]):
        """Save journey data to file"""
        # Update metadata
        data["journey_metadata"]["last_updated"] = datetime.now().isoformat()
        data["journey_metadata"]["total_conversations"] = sum(
            len(week.get("conversations", [])) for week in data["weekly_conversations"]
        )
        data["journey_metadata"]["total_weeks"] = len(data["weekly_conversations"])
        
        with open(self.journey_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def start_new_week(self, week_number: int, week_type: str, week_context: str, 
                      date_range: tuple = None, travel_destination: str = None,
                      plan_adherence: float = 0.5) -> None:
        """Start a new week in the journey log
        
        Args:
            week_number: Week number (1-35 for 8 months)
            week_type: Type of week (normal, travel, biomarker, etc.)
            week_context: Context description for the week
            date_range: (start_date, end_date) tuple
            travel_destination: Destination if it's a travel week
            plan_adherence: Adherence percentage for the week
        """
        with self.lock:
            journey_data = self._load_journey_data()
            
            # Calculate date range if not provided
            if date_range is None:
                # Assume journey started from file creation date
                start_date = datetime.fromisoformat(
                    journey_data["journey_metadata"]["patient_profile"]["start_date"]
                )
                week_start = start_date + timedelta(weeks=week_number - 1)
                week_end = week_start + timedelta(days=6)
                date_range = (week_start, week_end)
            
            # Create week separator
            week_data = {
                "week_separator": f"=== WEEK {week_number} ({week_type.upper()}) ===",
                "week_metadata": {
                    "week_number": week_number,
                    "week_type": week_type,
                    "date_range": {
                        "start": date_range[0].isoformat() if hasattr(date_range[0], 'isoformat') else str(date_range[0]),
                        "end": date_range[1].isoformat() if hasattr(date_range[1], 'isoformat') else str(date_range[1])
                    },
                    "context": week_context,
                    "travel_destination": travel_destination,
                    "plan_adherence": plan_adherence,
                    "week_started": datetime.now().isoformat()
                },
                "conversations": [],
                "biomarker_report": None,
                "weekly_summary": None
            }
            
            # Add to journey data
            journey_data["weekly_conversations"].append(week_data)
            self._save_journey_data(journey_data)
            
            print(f"📅 Started Week {week_number} ({week_type}) in journey log")
    
    def add_conversation(self, conversation_data: Dict[str, Any], 
                        week_number: int = None) -> str:
        """Add a conversation to the current or specified week
        
        Args:
            conversation_data: Complete conversation data from orchestrator
            week_number: Specific week to add to (uses current week if None)
            
        Returns:
            str: Path to the journey file
        """
        with self.lock:
            journey_data = self._load_journey_data()
            
            # Find the target week
            target_week = None
            if week_number is not None:
                # Find specific week
                for week in journey_data["weekly_conversations"]:
                    if week.get("week_metadata", {}).get("week_number") == week_number:
                        target_week = week
                        break
            else:
                # Use the last (current) week
                if journey_data["weekly_conversations"]:
                    target_week = journey_data["weekly_conversations"][-1]
            
            if target_week is None:
                # Create a default week if none exists
                self.start_new_week(
                    week_number=week_number or 1,
                    week_type="normal",
                    week_context="Default week - conversation added without explicit week setup"
                )
                journey_data = self._load_journey_data()
                target_week = journey_data["weekly_conversations"][-1]
            
            # Prepare conversation entry with timestamp
            conversation_entry = {
                "conversation_id": conversation_data.get("conversation_id", 
                                                       datetime.now().strftime("%Y%m%d_%H%M%S")),
                **copy.deepcopy(conversation_data)
            }
            
            # Add week info if provided in conversation data
            if "week_info" in conversation_data:
                conversation_entry["week_context"] = conversation_data["week_info"]
            
            # Add to the target week
            target_week["conversations"].append(conversation_entry)
            
            # Update metadata
            target_week["week_metadata"]["last_conversation"] = datetime.now().isoformat()
            target_week["week_metadata"]["total_conversations"] = len(target_week["conversations"])
            
            self._save_journey_data(journey_data)
            
            week_num = target_week.get("week_metadata", {}).get("week_number", "Unknown")
            conversation_count = len(target_week["conversations"])
            print(f"💬 Added conversation to Week {week_num} (Total: {conversation_count} conversations)")
            
            return self.journey_file_path
    
    def add_biomarker_report(self, report: str, week_number: int = None) -> None:
        """Add a biomarker report to the specified week
        
        Args:
            report: Biomarker report content
            week_number: Week to add report to (uses current week if None)
        """
        with self.lock:
            journey_data = self._load_journey_data()
            
            # Find target week
            target_week = None
            if week_number is not None:
                for week in journey_data["weekly_conversations"]:
                    if week.get("week_metadata", {}).get("week_number") == week_number:
                        target_week = week
                        break
            else:
                if journey_data["weekly_conversations"]:
                    target_week = journey_data["weekly_conversations"][-1]
            
            if target_week:
                target_week["biomarker_report"] = {
                    "report": report,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_journey_data(journey_data)
                
                week_num = target_week.get("week_metadata", {}).get("week_number", "Unknown")
                print(f"🔬 Added biomarker report to Week {week_num}")
    
    def complete_week(self, week_number: int = None, summary: str = None) -> None:
        """Mark a week as complete and add optional summary
        
        Args:
            week_number: Week to complete (uses current week if None)
            summary: Optional summary of the week
        """
        with self.lock:
            journey_data = self._load_journey_data()
            
            # Find target week
            target_week = None
            if week_number is not None:
                for week in journey_data["weekly_conversations"]:
                    if week.get("week_metadata", {}).get("week_number") == week_number:
                        target_week = week
                        break
            else:
                if journey_data["weekly_conversations"]:
                    target_week = journey_data["weekly_conversations"][-1]
            
            if target_week:
                target_week["week_metadata"]["completed"] = True
                
                if summary:
                    target_week["weekly_summary"] = {
                        "summary": summary,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Add week completion separator
                target_week["week_completion"] = f"=== END WEEK {target_week['week_metadata']['week_number']} ==="
                
                self._save_journey_data(journey_data)
                
                week_num = target_week.get("week_metadata", {}).get("week_number", "Unknown")
                conversation_count = len(target_week.get("conversations", []))
                print(f"✅ Completed Week {week_num} with {conversation_count} conversations")
    
    def get_journey_stats(self) -> Dict[str, Any]:
        """Get statistics about the current journey
        
        Returns:
            Dict containing journey statistics
        """
        journey_data = self._load_journey_data()
        
        total_conversations = 0
        total_weeks = len(journey_data["weekly_conversations"])
        completed_weeks = 0
        week_types = {}
        
        for week in journey_data["weekly_conversations"]:
            conversations_in_week = len(week.get("conversations", []))
            total_conversations += conversations_in_week
            
            if week.get("week_metadata", {}).get("completed"):
                completed_weeks += 1
            
            week_type = week.get("week_metadata", {}).get("week_type", "unknown")
            week_types[week_type] = week_types.get(week_type, 0) + 1
        
        return {
            "total_weeks": total_weeks,
            "completed_weeks": completed_weeks,
            "total_conversations": total_conversations,
            "average_conversations_per_week": total_conversations / total_weeks if total_weeks > 0 else 0,
            "week_type_distribution": week_types,
            "journey_file_path": self.journey_file_path,
            "last_updated": journey_data["journey_metadata"]["last_updated"]
        }
    
    def get_week_data(self, week_number: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific week
        
        Args:
            week_number: Week number to retrieve
            
        Returns:
            Week data or None if not found
        """
        journey_data = self._load_journey_data()
        
        for week in journey_data["weekly_conversations"]:
            if week.get("week_metadata", {}).get("week_number") == week_number:
                return week
        
        return None
    
    def export_journey_summary(self) -> str:
        """Export a human-readable summary of the journey
        
        Returns:
            Path to the generated summary file
        """
        journey_data = self._load_journey_data()
        stats = self.get_journey_stats()
        
        summary_lines = []
        summary_lines.append("# 8-Month Patient Journey Summary")
        summary_lines.append("=" * 60)
        
        # Patient info
        patient = journey_data["journey_metadata"]["patient_profile"]
        summary_lines.append(f"\n## Patient: {patient['name']}")
        summary_lines.append(f"- Age: {patient['age']}")
        summary_lines.append(f"- Occupation: {patient['occupation']}")
        summary_lines.append(f"- Location: {patient['location']}")
        summary_lines.append(f"- Chronic Condition: {patient['chronic_condition']}")
        summary_lines.append(f"- Journey Start: {patient['start_date']}")
        
        # Journey statistics
        summary_lines.append(f"\n## Journey Statistics")
        summary_lines.append(f"- Total Weeks: {stats['total_weeks']}")
        summary_lines.append(f"- Completed Weeks: {stats['completed_weeks']}")
        summary_lines.append(f"- Total Conversations: {stats['total_conversations']}")
        summary_lines.append(f"- Average Conversations/Week: {stats['average_conversations_per_week']:.1f}")
        summary_lines.append(f"- Last Updated: {stats['last_updated']}")
        
        # Week type distribution
        summary_lines.append(f"\n## Week Type Distribution")
        for week_type, count in stats['week_type_distribution'].items():
            summary_lines.append(f"- {week_type.capitalize()}: {count} weeks")
        
        # Week-by-week breakdown
        summary_lines.append(f"\n## Week-by-Week Breakdown")
        for week in journey_data["weekly_conversations"]:
            week_meta = week.get("week_metadata", {})
            week_num = week_meta.get("week_number", "Unknown")
            week_type = week_meta.get("week_type", "unknown")
            conversation_count = len(week.get("conversations", []))
            context = week_meta.get("context", "No context")
            
            summary_lines.append(f"\n### Week {week_num} ({week_type.capitalize()})")
            summary_lines.append(f"- Context: {context}")
            summary_lines.append(f"- Conversations: {conversation_count}")
            
            if week_meta.get("travel_destination"):
                summary_lines.append(f"- Travel Destination: {week_meta['travel_destination']}")
            
            if week.get("biomarker_report"):
                summary_lines.append(f"- Biomarker Report: Yes")
            
            if week_meta.get("completed"):
                summary_lines.append(f"- Status: Completed")
            else:
                summary_lines.append(f"- Status: In Progress")
        
        # Save summary
        summary_content = "\n".join(summary_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"journey_summary_{timestamp}.md"
        summary_path = os.path.join(self.log_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📄 Journey summary exported to: {summary_filename}")
        return summary_path


# Backwards compatibility function
def create_journey_logger(log_dir: str = "logs", journey_filename: str = None) -> JourneyLogger:
    """Create a new journey logger instance
    
    Args:
        log_dir: Directory for log files
        journey_filename: Custom filename for journey log
        
    Returns:
        JourneyLogger instance
    """
    return JourneyLogger(log_dir, journey_filename)
