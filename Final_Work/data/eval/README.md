# 📊 Evaluation Data Structure

## 🧹 **Cleanup Summary**

The `data/eval` folder has been cleaned up to eliminate duplicate files and consolidate evaluation data into a single, organized structure.

### **Files Removed (Duplicates)**
- ❌ `ragas_testset.json` - Generic test set (merged into `evaluation_questions.json`)
- ❌ `enhanced_testset.json` - Detailed test set (merged into `evaluation_questions.json`)
- ❌ `evaluation_report.json` - Summary report (merged into `evaluation_results.json`)

### **Files Kept (Consolidated)**
- ✅ `evaluation_questions.json` - **Single source of truth** for all evaluation questions and data
- ✅ `evaluation_results.json` - **Complete evaluation results** with metrics and performance data
- ✅ `ground_truth.json` - **Ground truth data** for evaluation validation
- ✅ `pinecone_chunk_samples.json` - **Pinecone chunk samples** for analysis and debugging

## 📁 **File Descriptions**

### **1. `evaluation_questions.json` - Single Source of Truth**
- **Purpose**: Contains all evaluation questions, contexts, answers, and ground truth
- **Content**: 8 comprehensive test cases with detailed financial data
- **Structure**: Each question includes:
  - `id`: Unique identifier
  - `type`: Question type (textual/table)
  - `client_id`: Client identifier
  - `question`: The evaluation question
  - `expected_answer_notes`: Guidelines for expected answers
  - `contexts`: 3 context passages for each question
  - `answer`: Expected answer for evaluation
  - `ground_truth`: Ground truth data for validation

### **2. `evaluation_results.json` - Complete Results**
- **Purpose**: Stores all evaluation results and performance metrics
- **Content**: 
  - Test cases with contexts and answers
  - Average metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - Individual test case results with detailed metrics
  - Performance analysis and statistics

### **3. `ground_truth.json` - Validation Data**
- **Purpose**: Provides ground truth data for evaluation validation
- **Content**: Validated ground truth information for each test case
- **Usage**: Used by evaluation framework to validate results

### **4. `pinecone_chunk_samples.json` - Analysis Data**
- **Purpose**: Contains samples of Pinecone chunks for analysis
- **Content**: 
  - Chunk samples with metadata
  - Statistics about chunk distribution
  - Text samples for debugging and analysis

## 🔄 **Data Flow**

```
evaluation_questions.json → Evaluation Framework → evaluation_results.json
           ↓
    ground_truth.json → Validation → Performance Metrics
           ↓
    pinecone_chunk_samples.json → Analysis & Debugging
```

## 📊 **Test Case Overview**

The evaluation system includes **8 comprehensive test cases** covering:

1. **Financial Highlights Summary** - Q1 2025 financial performance overview
2. **Revenue Analysis** - Detailed revenue figures and metrics from tables
3. **Operational Improvements** - Business developments and efficiency gains
4. **Financial Performance** - Key indicators and profitability metrics
5. **Business Segments** - Segment performance and strategic initiatives
6. **Table Analysis** - Financial table trends and patterns
7. **Comprehensive Summary** - Complete Q1 2025 report summary
8. **Specific Metrics** - Detailed numerical data extraction

## 🎯 **Benefits of Consolidation**

### **Before Cleanup**
- ❌ **4 duplicate files** with overlapping content
- ❌ **Confusion** about which file to use
- ❌ **Maintenance overhead** keeping files in sync
- ❌ **Storage waste** from duplicate data

### **After Cleanup**
- ✅ **Single source of truth** for evaluation data
- ✅ **Clear file purposes** and responsibilities
- ✅ **Reduced maintenance** and confusion
- ✅ **Optimized storage** and organization
- ✅ **Easier debugging** and analysis

## 🚀 **Usage Guidelines**

### **For Evaluation**
- Use `evaluation_questions.json` as the **input** for evaluation runs
- Reference `ground_truth.json` for **validation**
- Store results in `evaluation_results.json`

### **For Analysis**
- Use `evaluation_results.json` for **performance analysis**
- Reference `pinecone_chunk_samples.json` for **debugging**
- Use `evaluation_questions.json` for **question templates**

### **For Development**
- Modify `evaluation_questions.json` to **add new test cases**
- Update `ground_truth.json` when **data changes**
- Regenerate `evaluation_results.json` after **evaluation runs**

---

**Last Updated**: January 2025  
**Status**: ✅ **Clean and Organized**  
**Total Files**: 4 (down from 7)
