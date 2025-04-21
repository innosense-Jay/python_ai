# train_script.py
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import torch
import torch_directml  # ต้องติดตั้ง torch-directml

# ✅ ตั้งค่า Device
try:
    device = torch_directml.device()
    print(f"Device: {device}")
except:
    device = torch.device("cpu")
    print("⚠️ ใช้ CPU แทน")

# ✅ โหลดข้อมูล
def load_dataset_local(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [InputExample(
            texts=[json.loads(line)['text1'], json.loads(line)['text2']],
            label=float(json.loads(line)['label'])
        ) for line in f]

# ✅ ตั้งค่าโมเดล (แก้ไขสำคัญ!)
model = SentenceTransformer(
    "BAAI/bge-m3",
    device=device  # ตั้งค่า device ในโมเดลโดยตรง
)
model.to(device)  # ย้ายโมเดลไป GPU อีกครั้ง

# ✅ ตรวจสอบอุปกรณ์
print(f"\n=== ตรวจสอบการทำงานของ GPU ===")
print(f"โมเดลทำงานบน: {next(model.parameters()).device}")

# ✅ ปรับปรุง DataLoader
def collate_fn(batch):
    texts = []
    labels = []
    for example in batch:
        texts.append(example.texts)
        labels.append(example.label)
    
    # สร้าง Tensor และย้ายไป GPU
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    return {'texts': texts, 'labels': labels}

train_dataloader = DataLoader(
    load_dataset_local("train_data.jsonl"),
    shuffle=True,
    batch_size=8,
    collate_fn=collate_fn
)
# ✅ ตั้งค่าการฝึก
train_loss = losses.CosineSimilarityLoss(model)
val_evaluator = BinaryClassificationEvaluator.from_input_examples(
    load_dataset_local("val_data.jsonl"), 
    name="val-eval"
)
val_evaluator.model = model  # <-- ไม่มี error แล้ว

# ✅ เริ่มการฝึก
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=2,
    evaluation_steps=100,
    save_best_model=True,
    output_path="output_model_gpu",
    show_progress_bar=True
)

print("\n✅ บันทึกโมเดลไว้ที่: ./output_model_gpu")