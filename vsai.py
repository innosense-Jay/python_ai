
from sentence_transformers import SentenceTransformer, util
import pandas as pd

target_text = "iPhone 13 Pro Max สีเทาดำ มีรอยถลอกเล็กน้อยบริเวณฝาหลัง ใส่เคสยางสีแดงสด"
data = [
    {"name":"iphone 13 pro max สีขาว ไม่ใส่เคส ไม่ติดฟิลม์ ","index":2,"point":4},
    {"name":"iphone 13 pro max สีขาว ไม่ใส่เคส ไม่ติดฟิลม์ ","index":3,"point":4},
    {"name":"iPhone 13 Pro Max สีฟ้า รอยขีดข่วนที่มุมขวาล่าง เคสใส","index":19,"point":4},
    {"name":"iPhone 13 Pro Max สีเงิน รอยขีดข่วนที่ขอบจอ เคสซิลิโคนสีส้ม","index":32,"point":4},
    {"name":"iPhone 13 Pro Max สีฟ้า รอยขีดข่วนที่มุมซ้าย เคสยางสีแดง","index":33,"point":4},
    {"name":"iPhone 13 Pro Max สีทอง รอยถลอกที่มุมขวาบน เคสใส","index":34,"point":4},
    {"name":"iPhone 13 Pro Max สีเขียว รอยขีดข่วนที่มุมขวาล่าง เคสหนังสีเทา","index":35,"point":4},
    {"name":"iPhone 13 Pro Max สีเทา รอยขีดข่วนที่ขอบด้านซ้าย เคสหนังสีเทา","index":36,"point":4},
    {"name":"iPhone 13 Pro Max สีเทา รอยขีดข่วนที่มุมซ้ายบน เคสหนังสีส้ม","index":38,"point":4},
    {"name":"iPhone 13 Pro Max สีเทา รอยขีดข่วนเล็กน้อยที่มุมขวาล่าง เคสซิลิโคนสีฟ้า","index":39,"point":4},
    {"name":"iPhone 13 Pro Max สีเทา รอยขีดข่วนที่ขอบจอด้านซ้าย เคสยางสีดำ","index":40,"point":4},
    {"name":"iPhone 13 Pro Max สีเทา รอยถลอกที่ด้านหลัง เคสซิลิโคนสีแดง","index":41,"point":4},
    {"name":"iPhone 12 / 13 Pro Max  สีเทา รอยถลอกที่ด้านหลัง เคสซิลิโคนสีแดง","index":43,"point":4},
    {"name":"iPhone 13 Pro Max  สีเทา รอยถลอกที่ด้านหลัง เคสซิลิโคนสีแดง","index":44,"point":4},
    {"name":"iPhone 13 Pro Max  สีเทา  รอยด้านหลัง เคสสีแดง","index":45,"point":4},
    {"name":"iPhone 13 Pro Max  สีกราไฟท รอยด้านหลัง เคสสีแดง","index":46,"point":4}
]

# โหลดโมเดล
model_old = SentenceTransformer("BAAI/bge-m3")
model_new = SentenceTransformer("jaeyong2/bge-m3-Thai")

# แปลงข้อความ
target_emb_old = model_old.encode(target_text, convert_to_tensor=True)
target_emb_new = model_new.encode("query: " + target_text, convert_to_tensor=True)

# ประมวลผล
results = []
for entry in data:
    emb_old = model_old.encode(entry["name"], convert_to_tensor=True)
    emb_new = model_new.encode("passage: " + entry["name"], convert_to_tensor=True)
    score_old = util.cos_sim(target_emb_old, emb_old).item()
    score_new = util.cos_sim(target_emb_new, emb_new).item()
    results.append({
        "index": entry["index"],
        "name": entry["name"],
        "BGE_M3": round(score_old, 4),
        "E5_Large": round(score_new, 4)
    })

# แสดงผล
df = pd.DataFrame(results)
df = df.sort_values("E5_Large", ascending=False)
print(df.to_string(index=False))
