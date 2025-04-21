from sentence_transformers import SentenceTransformer, util
import json

# ✅ โหลดโมเดลที่ฝึกไว้
# model = SentenceTransformer("intfloat/multilingual-e5-small")  # เปลี่ยน path ได้
# model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")  # เปลี่ยน path ได้ 
model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1-6k3r")  # เปลี่ยน path ได้
# model = SentenceTransformer("BAAI/bge-m3")  # เปลี่ยน path ได้
# ✅ ข้อความเป้าหมาย (ของที่หาย)
target_text = "iPhone 13 Pro Max สีเทาดำ มีรอยถลอกเล็กน้อยบริเวณฝาหลัง ใส่เคสยางสีแดงสด"
# target_text = "iPhone 13 Pro Max gray-black, small scratch on back cover, wearing red rubber case"

target_emb = model.encode(target_text, convert_to_tensor=True)

# ✅ รายการของที่เจอ (จาก JSON)
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

# data = [
#     {"name": "iPhone 13 Pro Max white, no case, no film", "index": 2, "point": 4},
#     {"name": "iPhone 13 Pro Max white, no case, no film", "index": 3, "point": 4},
#     {"name": "iPhone 13 Pro Max blue, scratch on bottom right corner, clear case", "index": 19, "point": 4},
#     {"name": "iPhone 13 Pro Max silver, scratch on screen border, orange silicone case", "index": 32, "point": 4},
#     {"name": "iPhone 13 Pro Max blue, scratch on left corner, red rubber case", "index": 33, "point": 4},
#     {"name": "iPhone 13 Pro Max gold, scratch on top right corner, clear case", "index": 34, "point": 4},
#     {"name": "iPhone 13 Pro Max green, scratch on bottom right corner, gray leather case", "index": 35, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, scratch on left side, gray leather case", "index": 36, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, scratch on upper left corner, orange leather case", "index": 38, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, small scratch on bottom right corner, blue silicone case", "index": 39, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, scratch on left screen border, black rubber case", "index": 40, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, scratch on back, red silicone case", "index": 41, "point": 4},
#     {"name": "iPhone 12/13 Pro Max gray, scratch on back, red silicone case", "index": 43, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, scratch on back, red silicone case", "index": 44, "point": 4},
#     {"name": "iPhone 13 Pro Max gray, back scratch, red case", "index": 45, "point": 4},
#     {"name": "iPhone 13 Pro Max graphite, back scratch, red case", "index": 46, "point": 4}
# ]
# ✅ กำหนด threshold (จากโมเดลที่คุณฝึกเอง อาจใช้ ~0.80)
threshold = 0.85

# ✅ คำนวณคะแนนความเหมือน
results = []
for entry in data:
    candidate_emb = model.encode(entry["name"], convert_to_tensor=True)
    score = util.cos_sim(target_emb, candidate_emb).item()
    results.append({
        "index": entry["index"],
        "name": entry["name"],
        "score": score,
        "match": score >= threshold
    })

# ✅ เรียงจากมาก → น้อย
results = sorted(results, key=lambda x: x["score"], reverse=True)

# ✅ แสดงผล
print(f"\n🎯 ข้อความเป้าหมาย: {target_text}")
print(f"{'-'*100}")
for res in results:
    mark = "✅" if res["match"] else "❌"
    print(f"{mark} [index {res['index']:>2}] | Score: {res['score']:.4f} | {res['name']}")
