"""
API Key Management System for Rate Limiting and Load Balancing
Implements round-robin key rotation with weekly cycling for optimal rate limit management.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


class KeyStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class APIKeyInfo:
    """Information about an API key"""
    key: str
    nickname: str
    status: KeyStatus
    total_requests: int = 0
    weekly_requests: int = 0
    last_used: Optional[str] = None
    rate_limit_reset: Optional[str] = None
    error_count: int = 0
    created_date: str = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


@dataclass
class WeeklyKeyAssignment:
    """Weekly key assignment for load balancing"""
    week_number: int
    week_start_date: str
    week_end_date: str
    assigned_key_nickname: str
    primary_key: str
    backup_keys: List[str]
    requests_count: int = 0


class APIKeyManager:
    """
    Manages API keys with round-robin rotation and weekly cycling for load balancing.
    """
    
    def __init__(self, config_file: str = "api_keys_config.json"):
        self.config_file = config_file
        self.api_keys: List[APIKeyInfo] = []
        self.weekly_assignments: List[WeeklyKeyAssignment] = []
        self.current_week_assignment: Optional[WeeklyKeyAssignment] = None
        self.load_config()
    
    def collect_api_keys(self) -> None:
        """
        Interactive method to collect API keys from user.
        """
        print("🔑 API Key Collection for Load Balancing")
        print("=" * 50)
        print("Please provide your Groq API keys for round-robin load balancing.")
        print("Each week, a different key will be used to distribute the load.")
        print("Minimum 2 keys recommended for effective load balancing.\n")
        
        keys_input = input("Enter your API keys (comma-separated): ").strip()
        
        if not keys_input:
            raise ValueError("No API keys provided!")
        
        # Split and clean keys
        keys = [key.strip() for key in keys_input.split(',') if key.strip()]
        
        if len(keys) < 1:
            raise ValueError("At least 1 API key is required!")
        
        # Clear existing keys
        self.api_keys = []
        
        # Add each key with optional nickname
        for i, key in enumerate(keys):
            nickname = input(f"Enter nickname for key {i+1} (or press Enter for default): ").strip()
            if not nickname:
                nickname = f"Key_{i+1}"
            
            key_info = APIKeyInfo(
                key=key,
                nickname=nickname,
                status=KeyStatus.ACTIVE
            )
            self.api_keys.append(key_info)
            print(f"✅ Added key: {nickname}")
        
        print(f"\n✅ Successfully added {len(self.api_keys)} API keys!")
        self.save_config()
        self._plan_weekly_assignments()
    
    def load_config(self) -> None:
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Load API keys with enum conversion
                self.api_keys = []
                for key_data in data.get('api_keys', []):
                    # Convert status string back to enum
                    if 'status' in key_data and isinstance(key_data['status'], str):
                        key_data['status'] = KeyStatus(key_data['status'])
                    key_info = APIKeyInfo(**key_data)
                    self.api_keys.append(key_info)
                
                # Load weekly assignments
                self.weekly_assignments = [
                    WeeklyKeyAssignment(**assignment) for assignment in data.get('weekly_assignments', [])
                ]
                
                print(f"📁 Loaded {len(self.api_keys)} API keys from config file.")
                
            except Exception as e:
                print(f"⚠️ Error loading config: {e}")
                self.api_keys = []
                self.weekly_assignments = []
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        # Convert dataclasses to dict with enum handling
        api_keys_data = []
        for key in self.api_keys:
            key_dict = asdict(key)
            key_dict['status'] = key.status.value  # Convert enum to string
            api_keys_data.append(key_dict)
        
        config_data = {
            'api_keys': api_keys_data,
            'weekly_assignments': [asdict(assignment) for assignment in self.weekly_assignments],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Configuration saved to {self.config_file}")
    
    def _get_week_number(self, date: datetime) -> int:
        """Calculate week number since start of the year."""
        year_start = datetime(date.year, 1, 1)
        week_number = (date - year_start).days // 7 + 1
        return week_number
    
    def _get_week_dates(self, week_number: int, year: int = None) -> tuple:
        """Get start and end dates for a given week number."""
        if year is None:
            year = datetime.now().year
        
        year_start = datetime(year, 1, 1)
        week_start = year_start + timedelta(weeks=week_number - 1)
        week_end = week_start + timedelta(days=6)
        
        return week_start, week_end
    
    def _plan_weekly_assignments(self) -> None:
        """Plan weekly key assignments for the next 52 weeks using round-robin."""
        if not self.api_keys:
            return
        
        print("📅 Planning weekly key assignments...")
        
        # Clear existing assignments
        self.weekly_assignments = []
        
        # Plan for next 52 weeks
        current_date = datetime.now()
        current_week = self._get_week_number(current_date)
        
        for week_offset in range(52):
            week_num = current_week + week_offset
            week_start, week_end = self._get_week_dates(week_num)
            
            # Round-robin assignment
            primary_key_index = week_offset % len(self.api_keys)
            primary_key = self.api_keys[primary_key_index]
            
            # Get backup keys (other available keys)
            backup_keys = [
                key.key for i, key in enumerate(self.api_keys) 
                if i != primary_key_index and key.status == KeyStatus.ACTIVE
            ]
            
            assignment = WeeklyKeyAssignment(
                week_number=week_num,
                week_start_date=week_start.isoformat(),
                week_end_date=week_end.isoformat(),
                assigned_key_nickname=primary_key.nickname,
                primary_key=primary_key.key,
                backup_keys=backup_keys
            )
            
            self.weekly_assignments.append(assignment)
        
        print(f"✅ Planned assignments for {len(self.weekly_assignments)} weeks")
        self.save_config()
    
    def get_current_week_key(self) -> tuple[str, str]:
        """
        Get the API key for the current week.
        Returns: (api_key, key_nickname)
        """
        current_date = datetime.now()
        current_week = self._get_week_number(current_date)
        return self.get_key_for_week(current_week)
    
    def get_key_for_week(self, week_number: int) -> tuple[str, str]:
        """
        Get the API key for a specific week number.
        For journey synthesis, use this method with the journey week number.
        Returns: (api_key, key_nickname)
        """
        # For journey synthesis, we use round-robin based on journey week number
        # rather than calendar week number
        if not self.api_keys:
            raise RuntimeError("No API keys configured!")
        
        # Use round-robin assignment based on week number
        key_index = (week_number - 1) % len(self.api_keys)
        selected_key_info = self.api_keys[key_index]
        
        # Check if selected key is available
        if selected_key_info.status == KeyStatus.ACTIVE:
            return selected_key_info.key, selected_key_info.nickname
        
        # If selected key is not available, find next available key
        for i in range(len(self.api_keys)):
            next_key_index = (key_index + i + 1) % len(self.api_keys)
            next_key_info = self.api_keys[next_key_index]
            if next_key_info.status == KeyStatus.ACTIVE:
                print(f"⚠️ Primary key {selected_key_info.nickname} unavailable for week {week_number}, using: {next_key_info.nickname}")
                return next_key_info.key, next_key_info.nickname
        
        # No keys available
        raise RuntimeError(f"No available API keys for week {week_number}! All keys may be rate limited.")
    
    def _get_key_info(self, api_key: str) -> Optional[APIKeyInfo]:
        """Get key info object for a given API key."""
        for key_info in self.api_keys:
            if key_info.key == api_key:
                return key_info
        return None
    
    def record_api_usage(self, api_key: str, success: bool = True, error_message: str = None) -> None:
        """Record API usage for tracking and rate limiting."""
        key_info = self._get_key_info(api_key)
        if not key_info:
            return
        
        key_info.total_requests += 1
        key_info.weekly_requests += 1
        key_info.last_used = datetime.now().isoformat()
        
        if not success:
            key_info.error_count += 1
            if error_message and "rate limit" in error_message.lower():
                key_info.status = KeyStatus.RATE_LIMITED
                # Set rate limit reset time (estimate 1 hour)
                reset_time = datetime.now() + timedelta(hours=1)
                key_info.rate_limit_reset = reset_time.isoformat()
                print(f"🚫 Key {key_info.nickname} rate limited. Will reset at {reset_time.strftime('%H:%M')}")
        else:
            # Reset error count on successful request
            if key_info.error_count > 0:
                key_info.error_count = max(0, key_info.error_count - 1)
        
        # Update weekly assignment request count
        if self.current_week_assignment:
            self.current_week_assignment.requests_count += 1
        
        self.save_config()
    
    def check_and_reset_rate_limits(self) -> None:
        """Check and reset rate-limited keys if their cooldown period has passed."""
        current_time = datetime.now()
        
        for key_info in self.api_keys:
            if key_info.status == KeyStatus.RATE_LIMITED and key_info.rate_limit_reset:
                reset_time = datetime.fromisoformat(key_info.rate_limit_reset)
                if current_time >= reset_time:
                    key_info.status = KeyStatus.ACTIVE
                    key_info.rate_limit_reset = None
                    print(f"✅ Key {key_info.nickname} rate limit reset - now active")
    
    def get_next_available_key(self, exclude_key: str = None) -> tuple[str, str]:
        """
        Get the next available key, excluding the specified key.
        Returns: (api_key, key_nickname)
        """
        self.check_and_reset_rate_limits()
        
        available_keys = [
            key for key in self.api_keys 
            if key.status == KeyStatus.ACTIVE and key.key != exclude_key
        ]
        
        if not available_keys:
            raise RuntimeError("No available API keys! All keys may be rate limited.")
        
        # Select key with least recent usage
        selected_key = min(available_keys, key=lambda k: k.total_requests)
        return selected_key.key, selected_key.nickname
    
    def rotate_to_next_key(self, current_key: str) -> tuple[str, str]:
        """
        Rotate to the next available key when current key fails.
        Returns: (new_api_key, new_key_nickname)
        """
        print(f"🔄 Rotating away from current key due to rate limiting...")
        return self.get_next_available_key(exclude_key=current_key)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of API key usage."""
        current_week = self._get_week_number(datetime.now())
        
        summary = {
            "total_keys": len(self.api_keys),
            "active_keys": len([k for k in self.api_keys if k.status == KeyStatus.ACTIVE]),
            "rate_limited_keys": len([k for k in self.api_keys if k.status == KeyStatus.RATE_LIMITED]),
            "current_week": current_week,
            "current_week_assignment": None,
            "key_details": []
        }
        
        # Current week assignment
        for assignment in self.weekly_assignments:
            if assignment.week_number == current_week:
                summary["current_week_assignment"] = {
                    "primary_key": assignment.assigned_key_nickname,
                    "requests_this_week": assignment.requests_count
                }
                break
        
        # Key details
        for key_info in self.api_keys:
            summary["key_details"].append({
                "nickname": key_info.nickname,
                "status": key_info.status.value,
                "total_requests": key_info.total_requests,
                "weekly_requests": key_info.weekly_requests,
                "error_count": key_info.error_count,
                "last_used": key_info.last_used
            })
        
        return summary
    
    def print_usage_summary(self) -> None:
        """Print a formatted usage summary."""
        summary = self.get_usage_summary()
        
        print("\n📊 API Key Usage Summary")
        print("=" * 50)
        print(f"Total Keys: {summary['total_keys']}")
        print(f"Active Keys: {summary['active_keys']}")
        print(f"Rate Limited Keys: {summary['rate_limited_keys']}")
        print(f"Current Week: {summary['current_week']}")
        
        if summary['current_week_assignment']:
            assignment = summary['current_week_assignment']
            print(f"This Week's Primary Key: {assignment['primary_key']}")
            print(f"Requests This Week: {assignment['requests_this_week']}")
        
        print("\n🔑 Individual Key Status:")
        for key_detail in summary['key_details']:
            status_emoji = {
                'active': '✅',
                'rate_limited': '🚫', 
                'error': '❌',
                'disabled': '⏸️'
            }.get(key_detail['status'], '❓')
            
            print(f"{status_emoji} {key_detail['nickname']}: {key_detail['total_requests']} requests, {key_detail['error_count']} errors")
    
    def reset_weekly_counters(self) -> None:
        """Reset weekly request counters for all keys."""
        for key_info in self.api_keys:
            key_info.weekly_requests = 0
        
        print("🔄 Weekly request counters reset")
        self.save_config()


def ensure_key_manager_setup() -> APIKeyManager:
    """
    Ensure API key manager is set up. If no keys exist, prompt user to add them.
    """
    manager = APIKeyManager()
    
    if not manager.api_keys:
        print("🔑 No API keys found. Let's set them up!")
        manager.collect_api_keys()
    else:
        print(f"✅ Found {len(manager.api_keys)} configured API keys")
        manager.print_usage_summary()
    
    return manager


if __name__ == "__main__":
    # Test the key manager
    manager = ensure_key_manager_setup()
    manager.print_usage_summary()
