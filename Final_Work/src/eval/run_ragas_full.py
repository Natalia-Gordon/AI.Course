import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall
from src.pipeline.hybrid_rag_pipeline import HybridRAGPipeline
from pathlib import Path

EVAL_FILE = Path("data/eval/evaluation_questions_ayalon.json")
GROUND_TRUTH_FILE = Path("data/eval/ground_truth_ayalon.json")
REPORT_FILE = Path("data/eval/ayalon_ragas_full_report.json")

def load_eval():
    questions = json.loads(EVAL_FILE.read_text(encoding='utf-8')) if EVAL_FILE.exists() else []
    truths = json.loads(GROUND_TRUTH_FILE.read_text(encoding='utf-8')) if GROUND_TRUTH_FILE.exists() else []
    truths_map = {t['id']: t for t in truths}
    return questions, truths_map

def build_dataset(pipeline: HybridRAGPipeline, questions, truths_map):
    records = []
    for q in questions:
        qid = q['id']
        gt = truths_map.get(qid)
        if not gt:
            continue
        resp = pipeline.query(q['question'])
        answer = resp.get('answer','')
        contexts = [c.get('text','') for c in resp.get('contexts',[])]
        records.append({
            'question': q['question'],
            'answer': answer,
            'contexts': contexts,
            'ground_truth': gt.get('answer','')
        })
    return Dataset.from_list(records) if records else Dataset.from_list([])

def main():
    questions, truths_map = load_eval()
    pipeline = HybridRAGPipeline()
    ds = build_dataset(pipeline, questions, truths_map)
    if len(ds) == 0:
        print("No evaluation records found. Make sure data/eval files and processed chunks exist.")
        return
    res = evaluate(ds, metrics=[faithfulness, answer_correctness, context_precision, context_recall])
    # save results
    try:
        df = res.to_pandas()
        df.to_json(REPORT_FILE, orient='records', force_ascii=False)
        print("Saved full RAGAS report to", REPORT_FILE)
    except Exception as e:
        print("Could not convert results to pandas DataFrame:", e)
    print(res)

if __name__ == '__main__':
    main()