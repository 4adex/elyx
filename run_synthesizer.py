#!/usr/bin/env python3
"""
Run the 8-month patient journey synthesizer

This script will generate a complete 8-month patient journey simulation
with realistic constraints and interactions with the Elyx team.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.synthesizer import run_8_month_synthesis


async def main():
    print("🚀 Starting 8-Month Patient Journey Synthesis")
    print("This will generate realistic conversations over 8 months...")
    print("⚠️  Note: This will take a while as it generates ~175 conversations")
    print()
    
    # Confirm before starting
    response = input("Do you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Synthesis cancelled.")
        return
    
    try:
        # Run the synthesis
        output_dir = "synthesized_journey"
        summary_path = await run_8_month_synthesis(output_dir)
        
        print(f"\n✅ Synthesis complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Summary report: {summary_path}")
        
        # Show the summary
        print("\n" + "="*60)
        print("📋 JOURNEY SUMMARY")
        print("="*60)
        
        with open(summary_path, 'r') as f:
            print(f.read())
            
    except KeyboardInterrupt:
        print("\n⛔ Synthesis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
