#!/usr/bin/env python3
"""
Simple Graph Visualization Script for Medical Conversation Agent
Uses LangGraph's built-in Mermaid visualization capabilities.
"""

import sys
import os
from IPython.display import Image, display

# Add the src directory to the path so we can import our agent
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def visualize_medical_agent_graph():
    """Visualize the medical conversation agent graph using LangGraph's built-in capabilities"""
    
    try:
        # Import the medical conversation agent
        from src.medical_conversation_agent import MedicalConversationAgent
        
        print("🔧 Initializing Medical Conversation Agent...")
        
        # Create an instance of the agent
        agent = MedicalConversationAgent()
        
        print("📊 Generating graph visualization...")
        
        # Get the graph from the agent
        graph = agent.graph
        
        # Try to display the Mermaid diagram
        try:
            # This will display the graph as a Mermaid PNG
            display(Image(graph.get_graph().draw_mermaid_png()))
            print("✅ Graph visualization displayed successfully!")
            
        except Exception as viz_error:
            print(f"⚠️  Could not display PNG visualization: {viz_error}")
            print("📝 Trying to generate Mermaid text instead...")
            
            # Fallback: print the Mermaid text representation
            try:
                mermaid_text = graph.get_graph().draw_mermaid()
                print("\n" + "="*50)
                print("📋 MERMAID DIAGRAM (text representation)")
                print("="*50)
                print(mermaid_text)
                print("="*50)
                print("\n💡 You can copy this Mermaid text and paste it into:")
                print("   - https://mermaid.live/")
                print("   - Any Mermaid-compatible viewer")
                print("   - GitHub/GitLab markdown (```mermaid blocks)")
                
            except Exception as text_error:
                print(f"❌ Could not generate Mermaid text: {text_error}")
                
                # Final fallback: show graph structure information
                print("\n📊 Graph Structure Information:")
                print(f"   - Graph type: {type(graph)}")
                print(f"   - Available methods: {[method for method in dir(graph) if not method.startswith('_')]}")
    
    except ImportError as e:
        print(f"❌ Could not import medical conversation agent: {e}")
        print("💡 Make sure you're running this from the correct directory")
        print("   and that all dependencies are installed.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def save_mermaid_to_file(filename="medical_agent_graph.mmd"):
    """Save the Mermaid diagram to a file"""
    
    try:
        from src.medical_conversation_agent import MedicalConversationAgent
        
        agent = MedicalConversationAgent()
        graph = agent.graph
        
        mermaid_text = graph.get_graph().draw_mermaid()
        
        with open(filename, 'w') as f:
            f.write(mermaid_text)
            
        print(f"✅ Mermaid diagram saved to: {filename}")
        print(f"💡 You can view this file at: https://mermaid.live/")
        
        return filename
        
    except Exception as e:
        print(f"❌ Could not save Mermaid file: {e}")
        return None


def print_graph_info():
    """Print detailed information about the graph structure"""
    
    try:
        from src.medical_conversation_agent import MedicalConversationAgent
        
        agent = MedicalConversationAgent()
        graph = agent.graph
        
        print("\n" + "="*60)
        print("🏥 MEDICAL CONVERSATION AGENT - GRAPH ANALYSIS")
        print("="*60)
        
        # Get graph representation
        graph_repr = graph.get_graph()
        
        print(f"\n📊 Graph Type: {type(graph_repr)}")
        
        # Try to get nodes and edges information
        try:
            if hasattr(graph_repr, 'nodes'):
                print(f"\n🔹 Nodes ({len(graph_repr.nodes)}):")
                for i, node in enumerate(graph_repr.nodes, 1):
                    print(f"   {i}. {node}")
                    
            if hasattr(graph_repr, 'edges'):
                print(f"\n🔗 Edges ({len(graph_repr.edges)}):")
                for i, edge in enumerate(graph_repr.edges, 1):
                    print(f"   {i}. {edge}")
                    
        except Exception as e:
            print(f"⚠️  Could not extract nodes/edges: {e}")
            
        # Show available methods
        print(f"\n🛠️  Available Graph Methods:")
        methods = [method for method in dir(graph_repr) if not method.startswith('_')]
        for method in sorted(methods):
            print(f"   - {method}")
            
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Could not analyze graph: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Medical Conversation Agent Graph')
    parser.add_argument('--save', '-s', help='Save Mermaid diagram to file')
    parser.add_argument('--info', '-i', action='store_true', help='Show detailed graph information')
    parser.add_argument('--text-only', '-t', action='store_true', help='Only show text representation')
    
    args = parser.parse_args()
    
    if args.info:
        print_graph_info()
        
    if args.save:
        save_mermaid_to_file(args.save)
        
    if not args.info and not args.save:
        # Default behavior: try to visualize
        print("🚀 Medical Conversation Agent - Graph Visualizer")
        print("=" * 50)
        
        if args.text_only:
            # Only show text representation
            try:
                from src.medical_conversation_agent import MedicalConversationAgent
                agent = MedicalConversationAgent()
                mermaid_text = agent.graph.get_graph().draw_mermaid()
                print("\n📋 MERMAID DIAGRAM:")
                print("-" * 30)
                print(mermaid_text)
                print("-" * 30)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            visualize_medical_agent_graph()
