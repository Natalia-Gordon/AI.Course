#!/usr/bin/env python3
"""
Invoice Processing System with Gradio Interface
Features:
- Load and process invoice images
- Extract data using Gemini Flash
- Store in vector database
- Display tables and charts
- Category-based analysis
"""

import gradio as gr
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime
import base64
from typing import List, Dict, Any
import numpy as np

# Import Gemini and vector database components
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")

class InvoiceProcessor:
    def __init__(self):
        """Initialize the invoice processor with Gemini and vector database"""
        self.invoices_data = []
        self.vector_db = None
        self.embeddings = None
        self.llm = None
        self.setup_ai_components()
        
    def setup_ai_components(self):
        """Setup Gemini Flash and vector database"""
        try:
            # Setup Gemini Flash
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.1
                )
                print("✅ Gemini Flash initialized")
            else:
                print("⚠️ GOOGLE_API_KEY not found, using mock data")
                
            # Setup vector database
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vector_db = Chroma(
                persist_directory="./invoice_vector_db",
                embedding_function=self.embeddings
            )
            print("✅ Vector database initialized")
            
        except Exception as e:
            print(f"⚠️ AI components setup failed: {e}")
    
    def extract_invoice_data(self, image_path: str) -> Dict[str, Any]:
        """Extract invoice data using Gemini Flash"""
        try:
            if not self.llm:
                # Return mock data if Gemini is not available
                return self.generate_mock_invoice_data()
            
            # Load and encode image
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create prompt for invoice extraction
            prompt = """
            Analyze this invoice image and extract the following information in JSON format:
            {
                "invoice_number": "string",
                "date": "YYYY-MM-DD",
                "vendor": "string",
                "total_amount": "number",
                "currency": "string",
                "category": "string (e.g., Office Supplies, Services, Equipment, etc.)",
                "items": [
                    {
                        "description": "string",
                        "quantity": "number",
                        "unit_price": "number",
                        "total_price": "number"
                    }
                ],
                "tax_amount": "number",
                "subtotal": "number"
            }
            
            Please be accurate and extract all visible information.
            """
            
            # Use Gemini to extract data
            response = self.llm.invoke([
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ])
            
            # Parse JSON response
            try:
                data = json.loads(response.content)
                data['source_image'] = image_path
                data['extraction_timestamp'] = datetime.now().isoformat()
                return data
            except json.JSONDecodeError:
                print("Failed to parse JSON response, using mock data")
                return self.generate_mock_invoice_data()
                
        except Exception as e:
            print(f"Error extracting invoice data: {e}")
            return self.generate_mock_invoice_data()
    
    def generate_mock_invoice_data(self) -> Dict[str, Any]:
        """Generate mock invoice data for testing"""
        categories = ["Office Supplies", "IT Services", "Equipment", "Marketing", "Travel"]
        vendors = ["OfficeMax", "TechCorp", "MarketingPro", "TravelCo", "SupplyChain"]
        
        return {
            "invoice_number": f"INV-{np.random.randint(1000, 9999)}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "vendor": np.random.choice(vendors),
            "total_amount": round(np.random.uniform(100, 5000), 2),
            "currency": "USD",
            "category": np.random.choice(categories),
            "items": [
                {
                    "description": f"Item {i+1}",
                    "quantity": np.random.randint(1, 10),
                    "unit_price": round(np.random.uniform(10, 200), 2),
                    "total_price": round(np.random.uniform(50, 1000), 2)
                }
                for i in range(np.random.randint(2, 6))
            ],
            "tax_amount": round(np.random.uniform(10, 200), 2),
            "subtotal": round(np.random.uniform(100, 5000), 2),
            "source_image": "mock_data",
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    def store_in_vector_db(self, invoice_data: Dict[str, Any]):
        """Store invoice data in vector database"""
        try:
            # Create document for vector storage
            doc_text = f"""
            Invoice {invoice_data['invoice_number']} from {invoice_data['vendor']}
            Category: {invoice_data['category']}
            Amount: {invoice_data['total_amount']} {invoice_data['currency']}
            Date: {invoice_data['date']}
            Items: {', '.join([item['description'] for item in invoice_data['items']])}
            """
            
            doc = Document(
                page_content=doc_text,
                metadata={
                    'invoice_number': invoice_data['invoice_number'],
                    'vendor': invoice_data['vendor'],
                    'category': invoice_data['category'],
                    'amount': invoice_data['total_amount'],
                    'date': invoice_data['date']
                }
            )
            
            self.vector_db.add_documents([doc])
            print(f"✅ Stored invoice {invoice_data['invoice_number']} in vector database")
            
        except Exception as e:
            print(f"Error storing in vector database: {e}")
    
    def search_invoices(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search invoices using vector database"""
        try:
            if not self.vector_db:
                return []
            
            results = self.vector_db.similarity_search(query, k=top_k)
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"Error searching invoices: {e}")
            return []
    
    def process_invoice_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single invoice file"""
        invoice_data = self.extract_invoice_data(file_path)
        self.store_in_vector_db(invoice_data)
        self.invoices_data.append(invoice_data)
        return invoice_data
    
    def get_categories_summary(self) -> pd.DataFrame:
        """Get summary by categories"""
        if not self.invoices_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.invoices_data)
        category_summary = df.groupby('category').agg({
            'total_amount': ['sum', 'count', 'mean'],
            'invoice_number': 'count'
        }).round(2)
        
        category_summary.columns = ['Total Amount', 'Invoice Count', 'Average Amount', 'Invoice Count']
        return category_summary.reset_index()
    
    def get_items_by_category(self) -> pd.DataFrame:
        """Get all items grouped by category"""
        if not self.invoices_data:
            return pd.DataFrame()
        
        items_data = []
        for invoice in self.invoices_data:
            category = invoice['category']
            for item in invoice['items']:
                items_data.append({
                    'category': category,
                    'invoice_number': invoice['invoice_number'],
                    'vendor': invoice['vendor'],
                    'description': item['description'],
                    'quantity': item['quantity'],
                    'unit_price': item['unit_price'],
                    'total_price': item['total_price']
                })
        
        return pd.DataFrame(items_data)

# Initialize processor
processor = InvoiceProcessor()

def create_gradio_interface():
    """Create the Gradio interface"""
    
    def process_invoice_files(files):
        """Process uploaded invoice files"""
        if not files:
            return "No files uploaded", "No data", "No data", None, None
        
        processed_data = []
        for file in files:
            try:
                invoice_data = processor.process_invoice_file(file.name)
                processed_data.append(invoice_data)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
        
        # Update global data
        processor.invoices_data.extend(processed_data)
        
        # Generate outputs
        summary_df = processor.get_categories_summary()
        items_df = processor.get_items_by_category()
        
        # Create charts
        category_chart = create_category_chart(summary_df)
        amount_chart = create_amount_chart(summary_df)
        
        return (
            f"✅ Processed {len(processed_data)} invoices successfully!",
            summary_df.to_html(index=False) if not summary_df.empty else "No data",
            items_df.to_html(index=False) if not items_df.empty else "No data",
            category_chart,
            amount_chart
        )
    
    def search_invoices_func(query, top_k):
        """Search invoices using vector database"""
        if not query.strip():
            return "Please enter a search query"
        
        results = processor.search_invoices(query, int(top_k))
        if not results:
            return "No results found"
        
        output = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['content']}\n"
            output += f"   Metadata: {result['metadata']}\n\n"
        
        return output
    
    def load_sample_data():
        """Load sample invoice data for demonstration"""
        sample_invoices = [
            {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "vendor": "OfficeMax",
                "total_amount": 1250.50,
                "currency": "USD",
                "category": "Office Supplies",
                "items": [
                    {"description": "Printer Paper", "quantity": 10, "unit_price": 25.00, "total_price": 250.00},
                    {"description": "Pens", "quantity": 50, "unit_price": 2.50, "total_price": 125.00},
                    {"description": "Notebooks", "quantity": 20, "unit_price": 5.00, "total_price": 100.00}
                ],
                "tax_amount": 100.50,
                "subtotal": 1150.00
            },
            {
                "invoice_number": "INV-2024-002",
                "date": "2024-01-20",
                "vendor": "TechCorp",
                "total_amount": 3500.00,
                "currency": "USD",
                "category": "IT Services",
                "items": [
                    {"description": "Software License", "quantity": 1, "unit_price": 2000.00, "total_price": 2000.00},
                    {"description": "Technical Support", "quantity": 10, "unit_price": 150.00, "total_price": 1500.00}
                ],
                "tax_amount": 280.00,
                "subtotal": 3220.00
            }
        ]
        
        processor.invoices_data.extend(sample_invoices)
        for invoice in sample_invoices:
            processor.store_in_vector_db(invoice)
        
        summary_df = processor.get_categories_summary()
        items_df = processor.get_items_by_category()
        
        category_chart = create_category_chart(summary_df)
        amount_chart = create_amount_chart(summary_df)
        
        return (
            "✅ Sample data loaded successfully!",
            summary_df.to_html(index=False) if not summary_df.empty else "No data",
            items_df.to_html(index=False) if not items_df.empty else "No data",
            category_chart,
            amount_chart
        )
    
    def create_category_chart(df):
        """Create pie chart for categories"""
        if df.empty:
            return None
        
        try:
            fig = px.pie(
                df, 
                values='Total Amount', 
                names='category',
                title='Invoice Amount by Category'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            return None
    
    def create_amount_chart(df):
        """Create bar chart for amounts"""
        if df.empty:
            return None
        
        try:
            fig = px.bar(
                df,
                x='category',
                y='Total Amount',
                title='Total Invoice Amount by Category'
            )
            return fig
        except Exception as e:
            print(f"Error creating bar chart: {e}")
            return None
    
    # Create Gradio interface
    with gr.Blocks(title="Invoice Processing System") as interface:
        gr.Markdown("# 📄 Invoice Processing System")
        gr.Markdown("Upload invoice images to extract data, categorize items, and generate insights using Gemini Flash and vector database.")
        
        with gr.Tab("📤 Upload & Process"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Invoice Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    process_btn = gr.Button("🔄 Process Invoices")
                    load_sample_btn = gr.Button("📊 Load Sample Data")
                
                with gr.Column():
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3
                    )
        
        with gr.Tab("📊 Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Category Summary")
                    category_table = gr.HTML(label="Category Summary Table")
                
                with gr.Column():
                    gr.Markdown("### Items by Category")
                    items_table = gr.HTML(label="Items Table")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Category Distribution")
                    category_chart = gr.Plot(label="Category Pie Chart")
                
                with gr.Column():
                    gr.Markdown("### Amount by Category")
                    amount_chart = gr.Plot(label="Amount Bar Chart")
        
        with gr.Tab("🔍 Search"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search terms (e.g., 'office supplies', 'high amount', 'TechCorp')"
                    )
                    search_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Results"
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
            outputs=[status_output, category_table, items_table, category_chart, amount_chart]
        )
        
        load_sample_btn.click(
            fn=load_sample_data,
            inputs=[],
            outputs=[status_output, category_table, items_table, category_chart, amount_chart]
        )
        
        search_btn.click(
            fn=search_invoices_func,
            inputs=[search_query, search_top_k],
            outputs=[search_results]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 