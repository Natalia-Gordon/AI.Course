# 📄 Invoice Processing System

A comprehensive Gradio application for processing invoice images, extracting data using Gemini Flash, and generating insights with vector database search capabilities.

## 🚀 Features

- **📤 Upload & Process**: Upload invoice images and extract structured data
- **🤖 AI-Powered Extraction**: Uses Gemini Flash for intelligent invoice data extraction
- **🗄️ Vector Database**: Stores invoice data for semantic search
- **📊 Analytics**: Category-based analysis with interactive charts
- **🔍 Search**: Semantic search through invoice database
- **📋 Tables**: Detailed tables showing categories and items

## 🛠️ Installation

1. **Clone and navigate to the project:**
   ```bash
   cd Lessons/Lesson 15
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional):**
   ```bash
   # For Gemini Flash functionality
   export GOOGLE_API_KEY='your-google-api-key-here'
   
   # Or create a .env file:
   echo "GOOGLE_API_KEY=your-google-api-key-here" > .env
   ```

## 🎯 Usage

### Running the Application

```bash
python main.py
```

The application will launch at `http://localhost:7860` with a public shareable link.

### Using the Interface

#### 1. 📤 Upload & Process Tab
- **Upload Invoice Images**: Drag and drop or select invoice image files
- **Process Invoices**: Click to extract data from uploaded images
- **Load Sample Data**: Click to load demonstration data

#### 2. 📊 Analysis Tab
- **Category Summary Table**: Shows total amounts, counts, and averages by category
- **Items by Category Table**: Detailed breakdown of all items
- **Category Distribution Chart**: Pie chart showing category distribution
- **Amount by Category Chart**: Bar chart showing total amounts

#### 3. 🔍 Search Tab
- **Search Query**: Enter natural language queries
- **Number of Results**: Adjust how many results to return
- **Search Results**: View semantically similar invoices

## 📁 Project Structure

```
Lessons/Lesson 15/
├── main.py                 # Main entry point
├── invoice_processor.py    # Core processing logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── invoices/              # Sample invoice images
│   ├── invoice1.png
│   ├── invoice2.jpg
│   └── invoice3.png
└── invoice_vector_db/     # Vector database storage (auto-created)
```

## 🔧 Technical Details

### AI Components
- **Gemini Flash**: For invoice data extraction from images
- **Sentence Transformers**: For embedding generation
- **ChromaDB**: Vector database for semantic search

### Data Structure
Each processed invoice contains:
```json
{
  "invoice_number": "string",
  "date": "YYYY-MM-DD",
  "vendor": "string",
  "total_amount": "number",
  "currency": "string",
  "category": "string",
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
```

### Categories
The system automatically categorizes invoices into:
- Office Supplies
- IT Services
- Equipment
- Marketing
- Travel

## 🎨 Interface Features

### Upload & Process
- Multi-file upload support
- Real-time processing status
- Sample data loading for demonstration

### Analysis Dashboard
- Interactive HTML tables
- Plotly charts with hover information
- Responsive layout

### Search Functionality
- Natural language queries
- Semantic similarity search
- Configurable result count

## 🔍 Example Searches

Try these search queries:
- "office supplies"
- "high amount invoices"
- "TechCorp vendor"
- "recent invoices"
- "software licenses"

## 🚨 Troubleshooting

### Common Issues

1. **Gemini API Key Missing**
   - The system will use mock data
   - Set `GOOGLE_API_KEY` environment variable for full functionality

2. **Dependencies Not Found**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

3. **Port Already in Use**
   - Change the port in `main.py` line 25
   - Or kill the process using port 7860

4. **Vector Database Errors**
   - Delete the `invoice_vector_db/` folder to reset
   - Check disk space availability

### Performance Tips

- Use high-quality invoice images for better extraction
- Limit uploads to 10 images at once for optimal performance
- Close other applications to free up memory

## 🔮 Future Enhancements

- [ ] OCR fallback for text-based invoices
- [ ] Export functionality (CSV, Excel, PDF)
- [ ] Advanced filtering and sorting
- [ ] Multi-language support
- [ ] Invoice approval workflow
- [ ] Integration with accounting software

## 📄 License

This project is part of the AI Course curriculum.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Invoice Processing! 🎉** 