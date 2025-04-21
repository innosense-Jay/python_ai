from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import json

# ✅ โหลด test set
def load_dataset_local(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [InputExample(
            texts=[json.loads(line)['text1'], json.loads(line)['text2']],
            label=float(json.loads(line)['label'])
        ) for line in f]

test_examples = load_dataset_local("test_data.jsonl")

# ✅ สร้าง evaluator
test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_examples, name="test-eval")

# ✅ Before Test → ยังไม่ได้เทรน
print("\n🧪 Before Training Evaluation:")
model_before = SentenceTransformer("intfloat/multilingual-e5-small")
print(f"🔢 จำนวนตัวอย่างใน test set: {len(test_examples)}")

# ✅ ให้ evaluator print ผลลัพธ์เอง
eval_results = test_evaluator(model_before)
print("📊 Evaluation Results:", eval_results)

# ✅ After Test → โหลดโมเดลที่เทรนแล้ว
print("\n✅ After Training Evaluation:")
model_after = SentenceTransformer("output_model")  # หรือ path ที่คุณเซฟไว้
evalafter_results =test_evaluator(model_after)
print("📊 Evaluation Results:", evalafter_results)