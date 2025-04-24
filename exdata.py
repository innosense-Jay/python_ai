import pandas as pd
import json

# โหลดไฟล์ .parquet
file_path = '/Users/sathaporneuapattana/testapp/python_ai/0000.parquet'
df = pd.read_parquet(file_path)

# กรองข้อมูลที่มีคำว่า 'สี' ในคอลัมน์
filtered_df = df[df.apply(lambda row: row.astype(str).str.contains('สี').any(), axis=1)]

# กรองคำที่มี 'เสี' ออก เช่น 'เสียง', 'เสี่ยง'
filtered_df = filtered_df[~filtered_df.apply(lambda row: row.astype(str).str.contains('เสี').any(), axis=1)]

filtered_df = filtered_df[~filtered_df.apply(lambda row: row.astype(str).str.contains('สี่').any(), axis=1)]

filtered_df = filtered_df[~filtered_df.apply(lambda row: row.astype(str).str.contains('สีผิว').any(), axis=1)]

filtered_df = filtered_df[~filtered_df.apply(lambda row: row.astype(str).str.contains('สี่ช่อง').any(), axis=1)]

# เปลี่ยนชื่อคอลัมน์ให้ตรงกับรูปแบบที่ต้องการ
filtered_df = filtered_df.rename(columns={"context": "query", "Title": "pos", "Fake Title": "neg"})

# ฟังก์ชันสำหรับบันทึกข้อมูลเป็น JSONL
def save_as_jsonl(df, file_name):
    with open(file_name, "w", encoding="utf-8") as f:  # เพิ่มการกำหนดการเข้ารหัสเป็น utf-8
        for _, item in df.iterrows():  # ใช้ iterrows() เพื่อวนผ่านแถวใน DataFrame
            # แปลงแถวเป็น dictionary และแปลง pos, neg เป็น list
            item_dict = item.to_dict()
            item_dict['pos'] = [item_dict['pos']]  # เปลี่ยน pos เป็น list
            item_dict['neg'] = [item_dict['neg']]  # เปลี่ยน neg เป็น list
            f.write(f'{json.dumps(item_dict, ensure_ascii=False)}\n')  # ensure_ascii=False เพื่อให้จัดการกับตัวอักษรพิเศษ

# บันทึกข้อมูลเป็น JSONL
save_as_jsonl(filtered_df, "train_color.jsonl")

# แสดงผลลัพธ์
filtered_file_path = './filtered_data.parquet'
filtered_df.to_parquet(filtered_file_path)
print(f"ไฟล์ใหม่ถูกบันทึกที่: {filtered_file_path}")
