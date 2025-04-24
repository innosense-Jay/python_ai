from sentence_transformers import CrossEncoder
from pprint import pprint

# โหลดโมเดล
model = CrossEncoder("Pongsasit/mod-th-cross-encoder-minilm")

# คำถามหลัก
th_question = "iPhone 13 Pro Max สีเทาดำ มีรอยถลอกเล็กน้อยบริเวณฝาหลัง ใส่เคสยางสีแดงสด"

# ข้อมูลตัวอย่าง
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

# สร้างคู่ข้อความสำหรับการทำนาย
pairs = [[th_question, item["name"]] for item in data]

# ทำนายคะแนนความคล้าย
scores = model.predict(pairs)

# รวมผลลัพธ์กับข้อมูลเดิมและเรียงลำดับตามคะแนน
results = []
for item, score in zip(data, scores):
    result = item.copy()
    result["score"] = float(score)  # แปลง numpy float เป็น Python float
    results.append(result)

# เรียงลำดับจากคะแนนสูงสุดไปต่ำสุด
sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

# แสดงผลลัพธ์
print("คำถาม:", th_question)
print("\nผลลัพธ์ที่ตรงที่สุด 5 อันดับแรก:")
pprint(sorted_results[:5])

# หรือแสดงทั้งหมดแบบจัดรูปแบบ
print("\nผลลัพธ์ทั้งหมด:")
for idx, result in enumerate(sorted_results, 1):
    print(f"{idx}. {result['name']} (คะแนน: {result['score']:.4f}, index: {result['index']})")