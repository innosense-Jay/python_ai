# train_script.py
# ✅ ใช้สำหรับรันเทรนโมเดล BGE-M3 บนเครื่อง Windows (CPU / GPU)
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import os

# ✅ โหลดไฟล์ jsonl แล้วแปลงเป็น InputExample
def load_dataset_local(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [InputExample(
            texts=[json.loads(line)['text1'], json.loads(line)['text2']],
            label=float(json.loads(line)['label'])
        ) for line in f]

# ✅ กำหนด path ชุดข้อมูล
train_path = "train_data.jsonl"
val_path = "val_data.jsonl"

# ✅ โหลดข้อมูล
train_examples = load_dataset_local(train_path)
val_examples = load_dataset_local(val_path)

# ✅ โหลดโมเดล (bge-m3 จะโหลดจาก HuggingFace ครั้งแรก)
model = SentenceTransformer("BAAI/bge-m3")
# model = SentenceTransformer("intfloat/multilingual-e5-small")

# ✅ เตรียม Loss และ DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# ✅ เริ่มเทรน พร้อมประเมินผลระหว่างเทรน
from sentence_transformers.evaluation import BinaryClassificationEvaluator

val_evaluator = BinaryClassificationEvaluator.from_input_examples(val_examples, name="val-eval")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=2,
    evaluation_steps=100,
    save_best_model=True,
    output_path="output_model",
    show_progress_bar=True
)

print("\n✅ โมเดลถูกฝึกเสร็จ และบันทึกไว้ที่ ./output_model แล้ว")
