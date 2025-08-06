# Insurance PDF Analysis & RAGAS Evaluation

This project provides an automated workflow for analyzing insurance claim PDFs, generating concise summaries, and evaluating summary accuracy using the RAGAS metric.

## Features
- **PDF Summarization**: Extracts key facts, names, dates, and numbers from insurance documents using advanced LLM-based summarization.
- **RAGAS Evaluation**: Evaluates the factual accuracy of generated summaries against a ground truth using the RAGAS metric (LLM-based and fallback keyword matching).
- **Key Term Checking**: Automatically checks if all important terms from the ground truth are present in the summary.
- **Beautiful Log Output**: Saves formatted logs (including intermediate agent steps and final output) to the `output/` directory for easy review.

## Quickstart

1. **Install dependencies** (see requirements.txt for details):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your PDF**
   - Place your insurance claim PDF in the `data/` directory (e.g., `data/house_robbery_incident.pdf`).
   - (Optional) Place a plain text version as `data/house_robbery_incident.txt` for reference.

3. **Configure your ground truth**
   - Edit `main.py` and set the `ground_truth` variable to a concise, fact-rich summary (see example in the code).

4. **Run the main script**
   ```bash
   python Lessons/Lesson\ 10/Midlle\ Course/main.py
   ```

5. **Review the output**
   - The final output and all intermediate agent steps are printed to the terminal (with color highlighting).
   - A formatted log file is saved in `output/` (plain text, no color codes).
   - The log includes:
     - Timestamp
     - Intermediate agent steps
     - Final summary, contexts, and RAGAS score

## Helper Functions

- **Log Formatting**: `helper_functions.save_agent_log(response, plain_output, output_dir)`
  - Saves a readable log file with all steps and results.
- **Key Term Checking**: `helper_functions.check_key_terms(summary, ground_truth)`
  - Prints a warning if any important terms are missing from the summary.
- **Section Formatting**: `helper_functions.highlight_sections(text)` and `plain_sections(text)`
  - For colorized terminal output and clean log files.

## RAGAS Score Interpretation
- **Score > 0.8**: Summary is highly accurate and covers all key facts.
- **Score 0.5–0.8**: Some important details may be missing or phrasing may differ.
- **Score < 0.5**: Summary is incomplete or inaccurate; revise to include all key facts and match ground truth terminology.
- If the advanced RAGAS metric fails, a fallback keyword-matching score is used.

## Troubleshooting
- If you see warnings about missing key terms, revise your summary to include them.
- If you see output parsing errors, ensure the agent prompt and tool calls follow the documented format.
- For best results, keep the ground truth concise, fact-rich, and under 500 characters.

## License
MIT
