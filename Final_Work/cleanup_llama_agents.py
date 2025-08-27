#!/usr/bin/env python3
"""
Utility script to clean up existing LlamaExtract agents
Use this if you encounter conflicts with existing agent names
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def cleanup_llama_agents():
    """Clean up existing LlamaExtract agents to resolve conflicts."""
    try:
        from llama_cloud_services import LlamaExtract
        
        api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
        if not api_key:
            print("❌ LLAMA_CLOUD_API_KEY not found in .env file")
            return
        
        print("🔧 Initializing LlamaExtract...")
        llama_extract = LlamaExtract()
        
        # List existing agents
        print("\n📋 Existing extraction agents:")
        try:
            agents = llama_extract.list_agents()
            if agents:
                for agent in agents:
                    print(f"  - {agent.name} (ID: {agent.id})")
            else:
                print("  No agents found")
        except Exception as e:
            print(f"  Could not list agents: {e}")
        
        # Option to delete specific agents
        print("\n🗑️  To delete an agent, use the LlamaCloud dashboard:")
        print("   https://cloud.llamaindex.ai/")
        print("   Or use the API to delete specific agents")
        
        print("\n✅ Cleanup utility completed")
        print("💡 Tip: The system will now create agents with unique names")
        
    except ImportError:
        print("❌ llama-cloud-services not installed")
        print("   Install with: pip install llama-cloud-services")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    cleanup_llama_agents()
