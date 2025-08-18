#!/usr/bin/env python3
"""
Enhanced Journey Synthesizer Runner
This script provides an easy interface to run the enhanced synthesizer with API key management.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhanced_synthesizer import run_enhanced_8_month_synthesis
from src.key_manager import ensure_key_manager_setup


def print_banner():
    """Print a welcome banner."""
    print("=" * 70)
    print("🚀 Enhanced Elyx 8-Month Patient Journey Synthesizer")
    print("   with API Key Load Balancing & Rate Limit Management")
    print("=" * 70)
    print()


def print_features():
    """Print key features of the enhanced system."""
    print("🔧 Enhanced Features:")
    print("   ✅ Round-robin API key rotation (weekly)")
    print("   ✅ Automatic rate limit detection and recovery")
    print("   ✅ Load balancing across multiple API keys")
    print("   ✅ Real-time key usage tracking")
    print("   ✅ Intelligent key switching on failures")
    print("   ✅ Weekly assignment planning")
    print()


async def main():
    """Main runner function."""
    print_banner()
    print_features()
    
    try:
        # Setup API key manager
        print("🔑 Setting up API Key Management...")
        key_manager = ensure_key_manager_setup()
        
        # Show current configuration
        print("\n📊 Current Key Configuration:")
        key_manager.print_usage_summary()
        
        # Confirm to proceed
        proceed = input("\n🚀 Ready to start 8-month journey synthesis? (y/N): ").strip().lower()
        
        if proceed != 'y':
            print("👋 Synthesis cancelled by user.")
            return
        
        print("\n🎯 Starting Enhanced 8-Month Journey Synthesis...")
        print("   This will generate ~175 conversations (5 per week × 35 weeks)")
        print("   Keys will be rotated weekly and on rate limits")
        print("   Progress will be saved every 5 weeks")
        print()
        
        # Run the synthesis
        output_dir = "enhanced_synthesized_journey"
        summary_path = await run_enhanced_8_month_synthesis(output_dir)
        
        print(f"\n🎉 Synthesis Complete!")
        print(f"📄 Summary report: {summary_path}")
        print(f"📁 Full results in: {output_dir}/")
        
    except KeyboardInterrupt:
        print("\n⛔ Synthesis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
