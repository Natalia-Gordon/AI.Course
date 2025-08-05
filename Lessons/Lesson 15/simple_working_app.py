#!/usr/bin/env python3
"""
Invoice Processing System - Gradio 3.50.2 Compatible Version with Multiple File Upload
"""

import gradio as gr
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime

# Global storage for invoices
invoices_data = []

def generate_realistic_invoice_data():
    """Generate realistic invoice data"""
    categories = ["Office Supplies", "IT Services", "Equipment", "Marketing", "Travel", "Software", "Consulting"]
    vendors = ["OfficeMax", "TechCorp", "MarketingPro", "TravelCo", "SupplyChain", "SoftwareInc", "ConsultingPro"]
    
    invoice = {
        "invoice_number": f"INV-{np.random.randint(10000, 99999)}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "due_date": (datetime.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "vendor": np.random.choice(vendors),
        "vendor_address": f"{np.random.randint(100, 9999)} Main St, City, State {np.random.randint(10000, 99999)}",
        "customer": "Your Company Inc.",
        "customer_address": "123 Business Ave, Your City, State 12345",
        "total_amount": round(np.random.uniform(500, 10000), 2),
        "subtotal": round(np.random.uniform(400, 9000), 2),
        "tax_rate": 8.5,
        "tax_amount": round(np.random.uniform(30, 800), 2),
        "currency": "USD",
        "category": np.random.choice(categories),
        "payment_terms": "Net 30",
        "items": [
            {
                "description": f"Professional Service {i+1}",
                "quantity": np.random.randint(1, 5),
                "unit_price": round(np.random.uniform(50, 500), 2),
                "total_price": round(np.random.uniform(100, 2000), 2),
                "item_code": f"ITEM-{np.random.randint(1000, 9999)}"
            }
            for i in range(np.random.randint(2, 6))
        ],
        "notes": "Thank you for your business!",
        "processed_at": datetime.now().isoformat(),
        "confidence_score": round(np.random.uniform(0.85, 0.98), 3)
    }
    
    return invoice

def create_summary_table(invoices):
    """Create category summary table"""
    if not invoices:
        return "No data available"
    
    df = pd.DataFrame(invoices)
    summary = df.groupby('category').agg({
        'total_amount': ['sum', 'count', 'mean']
    }).round(2)
    
    summary.columns = ['Total Amount', 'Invoice Count', 'Average Amount']
    summary = summary.reset_index()
    
    return summary.to_html(index=False)

def create_items_table(invoices):
    """Create items by category table"""
    if not invoices:
        return "No data available"
    
    items_data = []
    for invoice in invoices:
        category = invoice['category']
        for item in invoice['items']:
            items_data.append({
                'category': category,
                'invoice_number': invoice['invoice_number'],
                'vendor': invoice['vendor'],
                'description': item['description'],
                'quantity': item['quantity'],
                'unit_price': item['unit_price'],
                'total_price': item['total_price'],
                'item_code': item.get('item_code', 'N/A')
            })
    
    df = pd.DataFrame(items_data)
    return df.to_html(index=False)

def create_dashboard_charts(invoices):
    """Create dashboard charts using HTML/CSS"""
    if not invoices:
        return "No data available for charts"
    
    df = pd.DataFrame(invoices)
    
    # Category distribution
    category_data = df.groupby('category')['total_amount'].sum()
    
    # Create HTML charts
    chart_html = f"""
    <div style="margin: 20px 0; font-family: Arial, sans-serif;">
        <h3 style="color: #333; text-align: center;">📊 Category Distribution</h3>
        <div style="display: flex; justify-content: center; align-items: center; gap: 40px; flex-wrap: wrap;">
            <!-- Pie Chart -->
            <div style="position: relative; width: 300px; height: 300px;">
                <svg width="300" height="300" viewBox="0 0 300 300">
    """
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43']
    
    # Calculate pie chart segments
    total_amount = category_data.sum()
    current_angle = 0
    
    for i, (category, amount) in enumerate(category_data.items()):
        percentage = (amount / total_amount) * 100
        angle = (amount / total_amount) * 360
        
        # Calculate SVG arc parameters
        start_angle_rad = math.radians(current_angle)
        end_angle_rad = math.radians(current_angle + angle)
        
        # Calculate start and end points
        start_x = 150 + 120 * math.cos(start_angle_rad)
        start_y = 150 + 120 * math.sin(start_angle_rad)
        end_x = 150 + 120 * math.cos(end_angle_rad)
        end_y = 150 + 120 * math.sin(end_angle_rad)
        
        # Determine if arc is large
        large_arc_flag = 1 if angle > 180 else 0
        
        # Create SVG path for pie slice
        color = colors[i % len(colors)]
        chart_html += f"""
                    <path d="M 150 150 L {start_x:.2f} {start_y:.2f} A 120 120 0 {large_arc_flag} 1 {end_x:.2f} {end_y:.2f} Z" 
                          fill="{color}" 
                          stroke="white" 
                          stroke-width="2"
                          style="cursor: pointer; transition: opacity 0.3s;"
                          onmouseover="this.style.opacity='0.8'"
                          onmouseout="this.style.opacity='1'"
                          title="{category}: ${amount:,.2f} ({percentage:.1f}%)"/>
        """
        
        # Add text labels on pie slices
        if percentage >= 5:  # Only show labels for slices >= 5%
            # Calculate position for text (middle of the slice)
            mid_angle = current_angle + (angle / 2)
            mid_angle_rad = math.radians(mid_angle)
            
            # Position text at 60% radius for better readability
            text_radius = 80
            text_x = 150 + text_radius * math.cos(mid_angle_rad)
            text_y = 150 + text_radius * math.sin(mid_angle_rad)
            
            # Determine text color (white for dark slices, dark for light slices)
            # Simple heuristic: use white for darker colors, dark for lighter colors
            color_hex = color.lstrip('#')
            r, g, b = int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "white" if brightness < 128 else "#333"
            
            # Add category name
            chart_html += f"""
                    <text x="{text_x:.2f}" y="{text_y - 5:.2f}" 
                          text-anchor="middle" 
                          dominant-baseline="middle"
                          fill="{text_color}"
                          font-family="Arial, sans-serif"
                          font-size="10"
                          font-weight="bold"
                          style="pointer-events: none;">
                        {category[:12]}{'...' if len(category) > 12 else ''}
                    </text>
            """
            
            # Add percentage
            chart_html += f"""
                    <text x="{text_x:.2f}" y="{text_y + 8:.2f}" 
                          text-anchor="middle" 
                          dominant-baseline="middle"
                          fill="{text_color}"
                          font-family="Arial, sans-serif"
                          font-size="9"
                          style="pointer-events: none;">
                        {percentage:.1f}%
                    </text>
            """
        
        current_angle += angle
    
    chart_html += """
                </svg>
            </div>
            
            <!-- Legend -->
            <div style="min-width: 250px;">
    """
    
    for i, (category, amount) in enumerate(category_data.items()):
        percentage = (amount / total_amount) * 100
        color = colors[i % len(colors)]
        chart_html += f"""
                <div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; border-radius: 8px; background: rgba(255,255,255,0.8); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; margin-right: 12px; flex-shrink: 0;"></div>
                    <div style="flex: 1;">
                        <div style="font-weight: bold; font-size: 14px; color: #333;">{category}</div>
                        <div style="font-size: 12px; color: #666;">${amount:,.2f} ({percentage:.1f}%)</div>
                    </div>
                </div>
        """
    
    chart_html += """
            </div>
        </div>
    </div>
    """
    
    # Vendor analysis
    vendor_data = df.groupby('vendor')['total_amount'].sum().sort_values(ascending=False)
    chart_html += f"""
    <div style="margin: 30px 0;">
        <h3 style="color: #333; text-align: center;">🏢 Vendor Analysis</h3>
        <div style="display: flex; flex-direction: column; gap: 15px; max-width: 600px; margin: 0 auto;">
    """
    
    for vendor, amount in vendor_data.items():
        chart_html += f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="font-size: 16px;">{vendor}</strong>
                <span style="font-size: 18px; font-weight: bold;">${amount:,.2f}</span>
            </div>
        </div>
        """
    
    chart_html += "</div></div>"
    
    # Summary statistics
    total_invoices = len(df)
    total_amount = df['total_amount'].sum()
    avg_amount = df['total_amount'].mean()
    
    chart_html += f"""
    <div style="margin: 30px 0;">
        <h3 style="color: #333; text-align: center;">📈 Summary Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 25px; max-width: 800px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 15px 0; font-size: 18px;">Total Invoices</h4>
                <p style="font-size: 32px; font-weight: bold; margin: 0;">{total_invoices}</p>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 15px 0; font-size: 18px;">Total Amount</h4>
                <p style="font-size: 32px; font-weight: bold; margin: 0;">${total_amount:,.2f}</p>
            </div>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 15px 0; font-size: 18px;">Average Amount</h4>
                <p style="font-size: 32px; font-weight: bold; margin: 0;">${avg_amount:,.2f}</p>
            </div>
        </div>
    </div>
    """
    
    return chart_html

def process_invoice_files(files):
    """Process multiple uploaded invoice files"""
    global invoices_data
    
    if not files:
        return "No files uploaded", "No data", "No data", "No data", "No data"
    
    try:
        processed_files = []
        new_invoices = []
        
        # Handle different file input formats from Gradio 3.50.2
        if isinstance(files, list):
            file_list = files
        elif hasattr(files, '__iter__') and not isinstance(files, str):
            file_list = list(files)
        else:
            file_list = [files]
        
        for file in file_list:
            if file is None:
                continue
                
            # Get file name - handle different file object structures
            if hasattr(file, 'name'):
                file_name = file.name
                file_path = file.name
            elif isinstance(file, (list, tuple)) and len(file) > 0:
                file_name = file[0].name if hasattr(file[0], 'name') else "uploaded_file"
                file_path = file[0].name if hasattr(file[0], 'name') else file[0]
            elif isinstance(file, str):
                import os
                file_name = os.path.basename(file)
                file_path = file
            else:
                file_name = "uploaded_file"
                file_path = str(file)
            
            # Process the actual uploaded file
            try:
                invoice_data = extract_invoice_data_from_file(file_path, file_name)
                if invoice_data:
                    invoice_data['source_file'] = file_name
                    new_invoices.append(invoice_data)
                    processed_files.append(file_name)
                else:
                    # Fallback to mock data if extraction fails
                    invoice_data = generate_realistic_invoice_data()
                    invoice_data['source_file'] = file_name
                    invoice_data['processing_note'] = "Mock data generated - file processing failed"
                    new_invoices.append(invoice_data)
                    processed_files.append(f"{file_name} (mock data)")
                    
            except Exception as e:
                # If file processing fails, generate mock data as fallback
                invoice_data = generate_realistic_invoice_data()
                invoice_data['source_file'] = file_name
                invoice_data['processing_note'] = f"Mock data generated - error: {str(e)}"
                new_invoices.append(invoice_data)
                processed_files.append(f"{file_name} (mock data)")
        
        # Add new invoices to existing data
        invoices_data.extend(new_invoices)
        
        # Create status message
        if len(processed_files) == 1:
            status_msg = f"✅ Processed 1 invoice file: {processed_files[0]}"
        else:
            status_msg = f"✅ Processed {len(processed_files)} invoice files: {', '.join(processed_files)}"
        
        # Convert to JSON string for display
        json_output = json.dumps(new_invoices, indent=2)
        
        # Create summary and items tables
        summary_table = create_summary_table(invoices_data)
        items_table = create_items_table(invoices_data)
        
        # Create dashboard charts
        chart_html = create_dashboard_charts(invoices_data)
        
        return (
            status_msg,
            json_output,
            summary_table,
            items_table,
            chart_html
        )
        
    except Exception as e:
        return f"❌ Error processing files: {str(e)}", "No data", "No data", "No data", "No data"

def extract_invoice_data_from_file(file_path, file_name):
    """Extract invoice data from uploaded file"""
    try:
        import os
        from PIL import Image
        import pytesseract
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None
            
        # Get file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Process based on file type
        if file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return extract_from_image(file_path, file_name)
        elif file_ext == '.pdf':
            return extract_from_pdf(file_path, file_name)
        else:
            return extract_from_text(file_path, file_name)
            
    except Exception as e:
        print(f"Error extracting data from {file_name}: {str(e)}")
        return None

def extract_from_image(file_path, file_name):
    """Extract invoice data from image files using OCR"""
    try:
        from PIL import Image
        import pytesseract
        
        # Open and process image
        image = Image.open(file_path)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        # Parse the extracted text to find invoice data
        invoice_data = parse_invoice_text(text, file_name)
        
        return invoice_data
        
    except Exception as e:
        print(f"Error processing image {file_name}: {str(e)}")
        return None

def extract_from_pdf(file_path, file_name):
    """Extract invoice data from PDF files"""
    try:
        import PyPDF2
        
        # Read PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Parse the extracted text
        invoice_data = parse_invoice_text(text, file_name)
        
        return invoice_data
        
    except Exception as e:
        print(f"Error processing PDF {file_name}: {str(e)}")
        return None

def extract_from_text(file_path, file_name):
    """Extract invoice data from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Parse the text
        invoice_data = parse_invoice_text(text, file_name)
        
        return invoice_data
        
    except Exception as e:
        print(f"Error processing text file {file_name}: {str(e)}")
        return None

def parse_invoice_text(text, file_name):
    """Parse extracted text to find invoice data"""
    try:
        import re
        from datetime import datetime
        
        # Initialize invoice data structure
        invoice_data = {
            "invoice_number": f"INV-{datetime.now().strftime('%Y%m%d')}-{hash(file_name) % 10000:04d}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "due_date": (datetime.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "vendor": "Unknown Vendor",
            "vendor_address": "Address not found",
            "customer": "Your Company Inc.",
            "customer_address": "123 Business Ave, Your City, State 12345",
            "total_amount": 0.0,
            "subtotal": 0.0,
            "tax_rate": 0.0,
            "tax_amount": 0.0,
            "currency": "USD",
            "category": "Uncategorized",
            "payment_terms": "Net 30",
            "items": [],
            "notes": "Processed from uploaded file",
            "processed_at": datetime.now().isoformat(),
            "confidence_score": 0.5,
            "extracted_text": text[:500] + "..." if len(text) > 500 else text
        }
        
        # Try to extract invoice number
        invoice_patterns = [
            r'invoice\s*#?\s*(\w+[-/]?\d+)',
            r'inv\s*#?\s*(\w+[-/]?\d+)',
            r'invoice\s*number\s*:?\s*(\w+[-/]?\d+)',
            r'(\d{4,6}[-/]\d{4,6})',
            r'(INV-\d+)'
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                invoice_data["invoice_number"] = match.group(1)
                break
        
        # Try to extract vendor name
        vendor_patterns = [
            r'from\s*:?\s*([A-Za-z\s&]+)',
            r'vendor\s*:?\s*([A-Za-z\s&]+)',
            r'company\s*:?\s*([A-Za-z\s&]+)',
            r'bill\s*to\s*:?\s*([A-Za-z\s&]+)'
        ]
        
        for pattern in vendor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vendor_name = match.group(1).strip()
                if len(vendor_name) > 2 and len(vendor_name) < 50:
                    invoice_data["vendor"] = vendor_name
                    break
        
        # Try to extract total amount
        amount_patterns = [
            r'total\s*:?\s*\$?([\d,]+\.?\d*)',
            r'amount\s*:?\s*\$?([\d,]+\.?\d*)',
            r'balance\s*due\s*:?\s*\$?([\d,]+\.?\d*)',
            r'\$([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*USD'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if amount > 0 and amount < 1000000:  # Reasonable range
                        invoice_data["total_amount"] = amount
                        invoice_data["subtotal"] = amount * 0.92  # Estimate
                        invoice_data["tax_amount"] = amount * 0.08  # Estimate
                        break
                except ValueError:
                    continue
        
        # Try to extract date
        date_patterns = [
            r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1)
                    # Try to parse the date
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            if len(parts[2]) == 2:
                                parts[2] = '20' + parts[2]
                            invoice_data["date"] = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    elif '-' in date_str:
                        invoice_data["date"] = date_str
                    break
                except:
                    continue
        
        # Determine category based on vendor name or text content
        categories = ["Office Supplies", "IT Services", "Equipment", "Marketing", "Travel", "Software", "Consulting"]
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['office', 'supply', 'paper', 'ink']):
            invoice_data["category"] = "Office Supplies"
        elif any(word in text_lower for word in ['tech', 'computer', 'it', 'software']):
            invoice_data["category"] = "IT Services"
        elif any(word in text_lower for word in ['equipment', 'hardware', 'device']):
            invoice_data["category"] = "Equipment"
        elif any(word in text_lower for word in ['marketing', 'advertising', 'promotion']):
            invoice_data["category"] = "Marketing"
        elif any(word in text_lower for word in ['travel', 'hotel', 'flight', 'transport']):
            invoice_data["category"] = "Travel"
        elif any(word in text_lower for word in ['software', 'license', 'subscription']):
            invoice_data["category"] = "Software"
        elif any(word in text_lower for word in ['consulting', 'service', 'professional']):
            invoice_data["category"] = "Consulting"
        
        # Generate items based on extracted data
        if invoice_data["total_amount"] > 0:
            invoice_data["items"] = [
                {
                    "description": f"Service from {invoice_data['vendor']}",
                    "quantity": 1,
                    "unit_price": invoice_data["total_amount"],
                    "total_price": invoice_data["total_amount"],
                    "item_code": f"ITEM-{hash(file_name) % 10000:04d}"
                }
            ]
        
        return invoice_data
        
    except Exception as e:
        print(f"Error parsing invoice text: {str(e)}")
        return None

def generate_sample_data():
    """Generate sample data for testing"""
    global invoices_data
    
    try:
        # Generate 5 sample invoices
        sample_invoices = []
        for i in range(5):
            invoice_data = generate_realistic_invoice_data()
            invoice_data['source_file'] = f"sample_invoice_{i+1}.pdf"
            sample_invoices.append(invoice_data)
        
        # Replace existing data with sample data
        invoices_data = sample_invoices
        
        json_output = json.dumps(invoices_data, indent=2)
        summary_table = create_summary_table(invoices_data)
        items_table = create_items_table(invoices_data)
        chart_html = create_dashboard_charts(invoices_data)
        
        return (
            f"✅ Generated {len(invoices_data)} sample invoices!",
            json_output,
            summary_table,
            items_table,
            chart_html
        )
        
    except Exception as e:
        return f"❌ Error generating sample data: {str(e)}", "No data", "No data", "No data", "No data"

def search_invoices(query):
    """Search through processed invoices"""
    global invoices_data
    
    if not query.strip():
        return "Please enter a search query"
    
    if not invoices_data:
        return "No invoices available to search"
    
    try:
        results = []
        for invoice in invoices_data:
            if query.lower() in invoice['vendor'].lower() or query.lower() in invoice['category'].lower():
                source_file = invoice.get('source_file', 'Unknown')
                results.append(f"Invoice {invoice['invoice_number']}: {invoice['vendor']} - {invoice['category']} - ${invoice['total_amount']} (from {source_file})")
        
        if results:
            return "\n".join(results)
        else:
            return "No results found"
            
    except Exception as e:
        return f"Error searching: {str(e)}"

def clear_data():
    """Clear all data"""
    global invoices_data
    invoices_data = []
    return "Data cleared", "No data", "No data", "No data", "No data"

# Create the main interface using Blocks for better control
with gr.Blocks(title="Invoice Processing System") as interface:
    gr.Markdown("# 📄 Invoice Processing System")
    gr.Markdown("Upload multiple invoice files, extract JSON data, and view interactive analysis with charts.")
    
    with gr.Tab("📤 Upload & Process"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Invoice Files (Multiple files supported)",
                    file_types=["image", "pdf"],
                    file_count="multiple"
                )
                with gr.Row():
                    process_btn = gr.Button("🔄 Process Invoices")
                    sample_btn = gr.Button("📊 Generate Sample Data")
                    clear_btn = gr.Button("🗑️ Clear Data")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
        
        with gr.Row():
            json_output = gr.Textbox(
                label="Extracted JSON Data",
                lines=15,
                interactive=False
            )
    
    with gr.Tab("📊 Analysis"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Category Summary")
                category_table = gr.HTML(label="Category Summary Table")
            
            with gr.Column():
                gr.Markdown("### Items by Category")
                items_table = gr.HTML(label="Items Table")
    
    with gr.Tab("📈 Interactive Dashboard"):
        gr.Markdown("### 📊 Interactive Analytics Dashboard")
        dashboard_charts = gr.HTML(label="Dashboard Charts")
    
    with gr.Tab("🔍 Search"):
        with gr.Row():
            with gr.Column():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter vendor name or category"
                )
                search_btn = gr.Button("🔍 Search")
            
            with gr.Column():
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=10,
                    interactive=False
                )
    
    # Event handlers
    process_btn.click(
        fn=process_invoice_files,
        inputs=[file_input],
        outputs=[status_output, json_output, category_table, items_table, dashboard_charts]
    )
    
    sample_btn.click(
        fn=generate_sample_data,
        inputs=[],
        outputs=[status_output, json_output, category_table, items_table, dashboard_charts]
    )
    
    clear_btn.click(
        fn=clear_data,
        inputs=[],
        outputs=[status_output, json_output, category_table, items_table, dashboard_charts]
    )
    
    search_btn.click(
        fn=search_invoices,
        inputs=[search_query],
        outputs=[search_results]
    )

if __name__ == "__main__":
    print("🚀 Starting Invoice Processing System with Multiple File Upload...")
    print("📄 Features:")
    print("   • Upload multiple invoice files at once")
    print("   • Extract JSON data from all invoices")
    print("   • View category-based analysis")
    print("   • Interactive dashboard with HTML charts")
    print("   • Search through processed invoices")
    print("   • Generate sample data for testing")
    print("   • Clear data functionality")
    print()
    
    # Launch with Gradio 3.50.2 - much more stable
    interface.launch(server_port=7864) 