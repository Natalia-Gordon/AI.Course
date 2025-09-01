# 🚀 Hybrid RAG System with Comprehensive Evaluation Framework

## 📋 Project Overview

This project implements a **Hybrid RAG (Retrieval-Augmented Generation) System** specifically designed for financial document analysis, featuring a comprehensive evaluation framework using RAGAS metrics. The system combines dense (Pinecone + OpenAI embeddings) and sparse (TF-IDF) retrieval with intelligent reranking and agent-based query processing.

## 🎯 Key Features

- **Hybrid Retrieval**: Combines dense and sparse retrieval methods
- **Multi-Agent Architecture**: Router, Summary, Needle, and Table QA agents
- **LlamaCloud Integration**: Structured data extraction for metadata enhancement
- **Comprehensive Evaluation**: RAGAS metrics with detailed performance analysis
- **Production-Ready Logging**: Unified logging system with component-specific tracking
- **Financial Document Focus**: Optimized for Q1 2025 financial reports and tables

## 🏗️ System Architecture

### Core Components

```
src/
├── core/           # Configuration and core utilities
├── agents/         # AI agents for different query types
├── ingest/         # Document ingestion and processing
├── index/          # Pinecone indexing and management
├── pipeline/       # Data processing pipeline
├── retrieve/       # Hybrid retrieval system
├── utils/          # Utility functions and logging
└── eval/           # Comprehensive evaluation framework
```

### Agent Architecture

1. **Router Agent**: LLM-based intent classification (summary, needle, table)
2. **Summary Agent**: Generates comprehensive summaries from retrieved content
3. **Needle Agent**: Finds specific information or answers precise questions
4. **Table QA Agent**: Handles quantitative analysis and table-related queries

### Retrieval System

- **Dense Retrieval**: OpenAI embeddings + Pinecone vector database
- **Sparse Retrieval**: TF-IDF with Hebrew-aware text processing
- **Reranking**: CrossEncoder with fallback scoring
- **Hybrid Fusion**: Intelligent combination of both retrieval methods

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- LlamaCloud API key (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Final_Work

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
LLAMACLOUD_API_KEY=your_llamacloud_api_key
```

### Basic Usage

```bash
# Run the main system
python src/main.py

# Run evaluation framework
python src/eval/run_evaluation.py

# Examine Pinecone chunks
python src/eval/examine_pinecone_chunks.py
```

## 📊 Evaluation Framework

### RAGAS Metrics

The system evaluates performance using industry-standard RAGAS metrics:

- **Context Precision** ≥ 0.75 (Target: 133.3% ✅)
- **Context Recall** ≥ 0.70 (Target: 125.0% ✅)
- **Faithfulness** ≥ 0.85 (Target: 100.8% ✅)
- **Answer Relevancy** ≥ 0.80 (Target: 91.8% ❌)

### Evaluation Components

1. **Ground Truth Manager**: Manages and validates ground truth data
2. **Test Set Generator**: Creates evaluation test cases
3. **Metrics Calculator**: Calculates RAGAS metrics using LangChain evaluators
4. **RAGAS Evaluator**: Orchestrates the evaluation process
5. **Results Display**: Comprehensive results table and analysis

### Running Evaluation

```bash
# Complete evaluation pipeline
python src/eval/run_evaluation.py

# View results
python src/eval/show_results.py
```

## 🔧 Configuration

### Main Configuration (`src/config.yaml`)

```yaml
# Document processing
documents_dir: data/documents
processed_dir: data/processed

# Embedding configuration
embedding:
  provider: openai
  model: text-embedding-3-small
  dim: 1536

# Pinecone configuration
pinecone:
  index_name: hybrid-rag

# Evaluation configuration
evaluation:
  ragas:
    targets:
      context_precision: 0.75
      context_recall: 0.70
      faithfulness: 0.85
      answer_relevancy: 0.80
```

### Logging Configuration

```yaml
logging:
  base_dir: logs
  levels:
    ragas_evaluation: INFO
    evaluation_runner: INFO
    ground_truth_manager: INFO
    metrics_calculator: INFO
    test_set_generator: INFO
  file_pattern: "{component}_{date}.log"
  rotation:
    max_size_mb: 10
    backup_count: 7
```

## 📁 Project Structure

```
Final_Work/
├── src/
│   ├── agents/                 # AI agents
│   │   ├── router.py          # Query routing
│   │   ├── summary_agent.py   # Summary generation
│   │   ├── needle_agent.py    # Specific information retrieval
│   │   └── table_qa_agent.py  # Table analysis
│   ├── core/                  # Core functionality
│   │   ├── config_manager.py  # Configuration management
│   │   └── data_loader.py     # Data loading utilities
│   ├── eval/                  # Evaluation framework
│   │   ├── run_evaluation.py  # Main evaluation runner
│   │   ├── ragas_evaluator.py # RAGAS evaluation
│   │   ├── metrics_calculator.py # Metrics calculation
│   │   ├── ground_truth_manager.py # Ground truth management
│   │   ├── test_set_generator.py # Test set generation
│   │   ├── show_results.py    # Results display
│   │   ├── config.py          # Evaluation configuration
│   │   └── examine_pinecone_chunks.py # Chunk examination
│   ├── ingest/                # Document ingestion
│   ├── index/                 # Indexing system
│   ├── pipeline/              # Data pipeline
│   ├── retrieve/              # Retrieval system
│   ├── utils/                 # Utilities
│   └── main.py                # Main application
├── data/                      # Data files
│   ├── documents/             # Source documents
│   ├── processed/             # Processed data
│   └── eval/                  # Evaluation data
├── logs/                      # Log files
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing and Validation

### Test Cases

The system includes 8 comprehensive test cases covering:
- Financial highlights and operational performance
- Revenue figures and financial metrics
- Operational improvements and business developments
- Financial performance indicators
- Business segments and performance
- Financial table analysis
- Comprehensive Q1 2025 summary
- Specific data and metrics

### Performance Metrics

- **Overall Score**: 0.867 (86.7%)
- **Targets Met**: 3/4 (75%)
- **Evaluation Strategy**: ragas_current_api
- **Total Test Cases**: 8

## 📈 Performance Analysis

### Current Performance

| Metric | Current | Target | Status | Performance |
|--------|---------|--------|--------|-------------|
| Context Precision | 1.000 | 0.75 | ✅ PASS | 133.3% |
| Context Recall | 0.875 | 0.70 | ✅ PASS | 125.0% |
| Faithfulness | 0.857 | 0.85 | ✅ PASS | 100.8% |
| Answer Relevancy | 0.735 | 0.80 | ❌ FAIL | 91.8% |

### Areas for Improvement

1. **Answer Relevancy**: Currently at 91.8% of target, needs improvement
2. **Test Case Optimization**: Focus on cases with lower performance
3. **Context Quality**: Maintain high context precision and recall

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Pinecone Connection**: Verify Pinecone environment and index configuration
3. **Evaluation Failures**: Check ground truth data and test set configuration
4. **Logging Issues**: Verify log directory permissions and configuration

### Debug Mode

Enable debug logging in `config.yaml`:

```yaml
logging:
  levels:
    ragas_evaluation: DEBUG
    evaluation_runner: DEBUG
```

## 🚀 Production Deployment

### Requirements

- **Memory**: Minimum 4GB RAM
- **Storage**: 10GB+ for documents and logs
- **API Limits**: Monitor OpenAI and Pinecone usage
- **Logging**: Configure log rotation and monitoring

### Monitoring

- **Performance Metrics**: Regular RAGAS evaluation runs
- **Log Analysis**: Monitor component performance and errors
- **API Usage**: Track OpenAI and Pinecone consumption
- **System Health**: Regular validation of ground truth and test sets

### Scaling

- **Horizontal Scaling**: Multiple instances for high availability
- **Load Balancing**: Distribute queries across instances
- **Caching**: Implement Redis for frequently accessed data
- **Async Processing**: Use Celery for background tasks

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RAGAS**: For the evaluation framework
- **LangChain**: For the evaluation integration
- **OpenAI**: For embedding and language models
- **Pinecone**: For vector database services

## 📞 Support

For questions, issues, or contributions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
