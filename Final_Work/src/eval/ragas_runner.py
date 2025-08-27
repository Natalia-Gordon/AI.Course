import json
from pathlib import Path
from typing import List, Dict

def load_testset(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def evaluate_stub(testset: List[Dict]) -> Dict[str, float]:
    return {
        "context_precision": 0.80,
        "context_recall": 0.72,
        "faithfulness": 0.88,
        "table_qa_accuracy": 0.92
    }
