#!/usr/bin/env python3
"""
Test script for real invoice processing functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_working_app import extract_invoice_data_from_file, parse_invoice_text

def test_text_parsing():
    """Test text parsing functionality"""
    print("🧪 Testing text parsing...")
    
    # Sample invoice text
    sample_text = """
    INVOICE #12345
    
    From: OfficeMax Supplies
    Date: 12/15/2024
    
    Total: $1,250.00
    
    Thank you for your business!
    """
    
    result = parse_invoice_text(sample_text, "test_invoice.txt")
    
    if result:
        print("✅ Text parsing successful!")
        print(f"   Invoice Number: {result['invoice_number']}")
        print(f"   Vendor: {result['vendor']}")
        print(f"   Amount: ${result['total_amount']}")
        print(f"   Date: {result['date']}")
        print(f"   Category: {result['category']}")
    else:
        print("❌ Text parsing failed!")
    
    return result

def test_file_processing():
    """Test file processing functionality"""
    print("\n🧪 Testing file processing...")
    
    # Create a test text file
    test_content = """
    INVOICE #TEST-001
    
    From: TechCorp Solutions
    Date: 01/20/2024
    
    Services Rendered:
    - IT Consulting: $2,500.00
    - Software License: $500.00
    
    Total: $3,000.00
    
    Payment Terms: Net 30
    """
    
    test_file_path = "test_invoice.txt"
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    try:
        result = extract_invoice_data_from_file(test_file_path, "test_invoice.txt")
        
        if result:
            print("✅ File processing successful!")
            print(f"   Invoice Number: {result['invoice_number']}")
            print(f"   Vendor: {result['vendor']}")
            print(f"   Amount: ${result['total_amount']}")
            print(f"   Category: {result['category']}")
        else:
            print("❌ File processing failed!")
        
        # Clean up
        os.remove(test_file_path)
        
    except Exception as e:
        print(f"❌ File processing error: {str(e)}")
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
    
    return result

if __name__ == "__main__":
    print("🚀 Starting Real Invoice Processing Tests...")
    print("=" * 50)
    
    # Test text parsing
    text_result = test_text_parsing()
    
    # Test file processing
    file_result = test_file_processing()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Text Parsing: {'✅ PASS' if text_result else '❌ FAIL'}")
    print(f"   File Processing: {'✅ PASS' if file_result else '❌ FAIL'}")
    
    if text_result and file_result:
        print("\n🎉 All tests passed! Real invoice processing is working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.") 