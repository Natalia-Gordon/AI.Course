#!/usr/bin/env python3
"""
Main entry point for the Invoice Processing System
"""

import os
from dotenv import load_dotenv
from invoice_processor import create_gradio_interface

def main():
    """Main function to run the invoice processing system"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for Google API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found in environment variables.")
        print("   The system will use mock data for demonstration.")
        print("   To use Gemini Flash, set your API key:")
        print("   export GOOGLE_API_KEY='your-api-key-here'")
        print()
    
    print("🚀 Starting Invoice Processing System...")
    print("📄 Features:")
    print("   • Upload and process invoice images")
    print("   • Extract data using Gemini Flash (if API key available)")
    print("   • Store data in vector database for search")
    print("   • Generate category-based analysis and charts")
    print("   • Search invoices using semantic similarity")
    print()
    
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    
    print("🌐 Launching Gradio interface...")
    interface.launch(
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()
