# Hybrid RAG System - Final Work

A comprehensive Metadata-driven Hybrid RAG (Retrieval-Augmented Generation) system for financial document analysis.



## 🚀 **Enhanced Integration: LangChain + LlamaExtract**

Your system now has **integrated LangChain and LlamaExtract enhancements** while maintaining the exact same architecture:

### **How It Works:**
1. **Router Agent** receives query and determines intent (summary, needle, or table) ← **UNCHANGED**
2. **Summary Agent** generates summaries using LangChain's map-reduce chains ← **ENHANCED**
3. **Needle Agent** extracts information using LangChain's custom prompts ← **ENHANCED**  
4. **Table QA Agent** analyzes tables with LangChain's specialized prompts ← **ENHANCED**
5. **Data Loader** automatically extracts structured financial data using LlamaExtract ← **INTEGRATED BY DEFAULT**

### **Usage Options:**

```bash
# 1. Standard Mode (default) - Now includes LlamaExtract enhancement automatically
python src/main.py --query "Summarize financial highlights"

# 2. LangChain Enhanced Mode
python src/main.py --query "Show me tables" --langchain

# 3. With verbose logging
python src/main.py --query "Test query" --verbose

# 4. Reprocess documents
python src/main.py --query "Test query" --reprocess
```

### **Benefits of Enhanced Integration:**
- **Better Summaries**: Uses LangChain's map-reduce chains for comprehensive summaries
- **Enhanced Extraction**: Custom prompts for precise information retrieval
- **Improved Table Analysis**: Specialized prompts for quantitative analysis
- **Automatic Financial Data Extraction**: LlamaExtract is now integrated into DataLoader by default
- **Enhanced Metadata**: Rich financial metrics, KPIs, and insights automatically extracted
- **Automatic Fallback**: Falls back to your existing agents if enhancements fail
- **Zero Architecture Changes**: Your Router → Agent flow remains exactly the same
- **Single Entry Point**: Everything integrated into one main.py file
- **Simplified Usage**: No additional flags needed - LlamaExtract enhancement is automatic

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Processing    │    │   Indexing      │
│   Ingestion     │───▶│   Pipeline      │───▶│   (Dense+Sparse)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query         │    │   Hybrid        │    │   Agent         │
│   Processing    │◀───│   Retrieval     │◀───│   Routing       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Reranking     │
                       │   (Cross-Encoder)│
                       └─────────────────┘
```

## 📁 Project Structure

```
Final_Work/
├── src/                          # Core system code
│   ├── main.py                  # Main entry point with logging
│   ├── config.yaml              # Configuration file
│   ├── ingest/                  # Document processing
│   │   ├── data_loader.py      # Data loading and management
│   │   ├── parse_pdf.py        # PDF text extraction
│   │   ├── chunking.py         # Document chunking
│   │   ├── metadata.py         # Metadata extraction
│   │   ├── entity_extractor.py # Entity recognition
│   │   └── table_processor.py  # Table processing
│   ├── index/                   # Indexing components
│   │   ├── pinecone_index.py   # Pinecone vector database
│   │   └── tfidf_index.py      # TF-IDF sparse index
│   ├── retrieve/                # Retrieval components
│   │   ├── hybrid.py           # Hybrid retriever
│   │   └── rerank.py           # Cross-encoder reranker
│   ├── agents/                  # AI agents
│   │   ├── router.py           # Query routing
│   │   ├── summary_agent.py    # Summary generation
│   │   ├── needle_agent.py     # Information extraction
│   │   └── table_qa_agent.py   # Table analysis
│   ├── eval/                    # Evaluation tools
│   │   └── ragas_evaluator.py  # RAGAS evaluation
│   ├── pipeline/                # Pipeline components
│   └── utils/                   # Utility functions
├── data/                        # Data directory
│   ├── documents/               # Source documents
│   │   └── ayalon_q1_2025.pdf  # Sample financial document
│   └── processed/               # Processed data
│       ├── chunks/              # Document chunks
│       ├── tables/              # Extracted tables
│       └── figures/             # Extracted figures
├── logs/                        # System logs
├── test_system.py               # System testing with logging
├── setup_environment.py         # Environment validation
├── cleanup_system.py            # System cleanup and validation
├── run_evaluation.py            # RAGAS evaluation runner
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Pinecone account and API key
- OpenAI account and API key

### 2. Install Dependencies
```bash
cd Final_Work
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the `AI.Course` directory with:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=us-east-1
PINECONE_CLOUD=aws
PINECONE_METRIC=cosine
OPENAI_API_KEY=your_openai_api_key
```

### 4. Validate Environment
```bash
python setup_environment.py
```

### 5. System Cleanup & Validation
```bash
python cleanup_system.py
```

## 🧪 Testing

### Run System Tests
```bash
python test_system.py
```

This will test:
- Environment variables
- PDF parsing
- Document chunking
- Entity extraction
- Table processing
- Data loader functionality

### Test Individual Components
```bash
# Test PDF parsing
python -c "from src.ingest.parse_pdf import extract_text_blocks; print('PDF parsing works')"

# Test entity extraction
python -c "from src.ingest.entity_extractor import EntityExtractor; print('Entity extraction works')"
```

## 🚀 Usage

### Basic Query
```bash
python src/main.py --query "Summarize the financial highlights"
```

### Advanced Options
```bash
python src/main.py \
  --query "What are the Q1 2025 revenue figures?" \
  --document "ayalon_q1_2025.pdf" \
  --reprocess \
  --verbose
```

### Query Types Supported

#### Summary Queries
- "Summarize the financial highlights"
- "Give me an executive summary"
- "What are the key findings?"

#### Needle Queries (Specific Information)
- "What is the revenue for Q1 2025?"
- "Who is the CEO?"
- "What was the profit margin?"

#### Table Queries (Quantitative Analysis)
- "Show me the revenue table"
- "Compare Q1 vs Q4 performance"
- "What are the key financial metrics?"

## 📊 Evaluation

### RAGAS Evaluation
```bash
python run_evaluation.py
```

Target Metrics:
- **Context Precision**: ≥0.75
- **Context Recall**: ≥0.70
- **Faithfulness**: ≥0.85
- **Answer Relevancy**: ≥0.80
- **Context Relevancy**: ≥0.80

### Custom Evaluation
```python
from src.eval.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()
results = evaluator.run_full_evaluation(questions, contexts, answers, ground_truth)
```

## 🔧 Configuration

### Main Configuration (`src/config.yaml`)
```yaml
project_name: "Hybrid RAG System"
embedding:
  model: "text-embedding-3-small"
  dim: 1536

pinecone:
  index_name: "hybrid-rag"
  metric: "cosine"
  region: "us-east-1"
  cloud: "aws"
  namespace: "C123"

retrieval:
  dense_k: 10
  sparse_k: 10
  final_k: 8
```

### Environment Variables
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_REGION`: Pinecone region (e.g., us-east-1)
- `PINECONE_CLOUD`: Cloud provider (e.g., aws)
- `OPENAI_API_KEY`: Your OpenAI API key

## 📝 Logging

The system uses comprehensive logging with:
- **File logging**: `hybrid_rag.log` for main system
- **Test logging**: `test_system.log` for testing
- **Structured format**: Timestamp, module, level, message
- **Verbose mode**: `--verbose` flag for debug information

### Log Levels
- **INFO**: General system information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors
- **DEBUG**: Detailed debugging (verbose mode)

## 🐛 Troubleshooting

### Common Issues

#### Environment Variables Not Loading
```bash
# Check if .env file exists in AI.Course directory
ls -la ../.env

# Validate environment
python setup_environment.py
```

#### Missing Dependencies
```bash
# Install missing packages
pip install -r requirements.txt

# Check specific package
python -c "import pinecone; print('Pinecone OK')"
```

#### PDF Processing Issues
```bash
# Check PDF file exists
ls -la data/documents/

# Test PDF parsing
python test_system.py
```

### Debug Mode
```bash
# Enable verbose logging
python src/main.py --query "test" --verbose

# Check logs
tail -f hybrid_rag.log
```

## 🔄 Development Workflow

### 1. Environment Setup
```bash
python setup_environment.py
```

### 2. System Validation
```bash
python cleanup_system.py
```

### 3. Testing
```bash
python test_system.py
```

### 4. Development
```bash
# Make changes to code
# Test changes
python test_system.py

# Run main system
python src/main.py --query "test query"
```

### 5. Evaluation
```bash
python run_evaluation.py
```

## 📚 API Reference

### Core Classes

#### DataLoader
```python
from src.ingest.data_loader import DataLoader

loader = DataLoader(config)
chunks = loader.load_document("document.pdf")
```

#### HybridRetriever
```python
from src.retrieve.hybrid import HybridRetriever

retriever = HybridRetriever(dense_index, sparse_index, reranker)
results = retriever.search("query", k_dense=10, k_sparse=10)
```

#### EntityExtractor
```python
from src.ingest.entity_extractor import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract_entities("text content")
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Run validation**: `python cleanup_system.py`
6. **Submit a pull request**

## 📄 License

This project is part of the AI Course final work. Please refer to the course guidelines for usage and distribution.

## 🙏 Acknowledgments

- **Pinecone**: Vector database infrastructure
- **OpenAI**: Embedding and language models
- **RAGAS**: Evaluation framework
- **Course Instructors**: Project guidance and requirements

---

**Last Updated**: 2025
**Version**: 1.0.0
**Status**: Ready for Production Use
