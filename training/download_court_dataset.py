import os
import sys
import zipfile
import requests

sys.stdout.reconfigure(encoding='utf-8')

zip_url = "https://app.roboflow.com/ds/mEZ0pCWOwb?key=tY3k14nJMl"
target_dir = os.path.abspath("training/datasets/court/tennis_court_keypoints")
zip_path = os.path.abspath("training/datasets/court/court_dataset.zip")

os.makedirs(target_dir, exist_ok=True)

print("=" * 60)
print("TẢI BỘ DATASET TENNIS COURT KEYPOINTS (10,000 ẢNH - 500 MB)")
print(f"URL: {zip_url}")
print(f"Lưu tạm: {zip_path}")
print(f"Giải nén vào: {target_dir}")
print("=" * 60)

print("\n[1/3] Đang tải file zip từ Roboflow (500 MB)...")
response = requests.get(zip_url, stream=True)
total_length = response.headers.get('content-length')

with open(zip_path, 'wb') as f:
    if total_length is None:
        f.write(response.content)
    else:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            dl += len(data)
            f.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write(f"\rTiến độ: [{'=' * done}{' ' * (50 - done)}] {dl / (1024*1024):.1f} MB / {total_length / (1024*1024):.1f} MB")
            sys.stdout.flush()

print("\n\n[2/3] Đang giải nén file zip...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

if os.path.exists(zip_path):
    os.remove(zip_path)
print("✅ Giải nén hoàn tất!")

print("\n[3/3] Kiểm tra các thư mục giải nén được:")
for item in os.listdir(target_dir):
    p = os.path.join(target_dir, item)
    if os.path.isdir(p):
        print(f" - Thư mục: {item} ({len(os.listdir(p))} files/subfolders)")
    else:
        print(f" - File: {item}")

# Chạy tạo data.json cho TensorFlow Keras
import glob, json

print("\n🔄 Đang tạo file data.json phục vụ TensorFlow Keras training...")
json_records = []
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(target_dir, split, 'images')
    lbl_dir = os.path.join(target_dir, split, 'labels')
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        continue
        
    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_p):
            continue
            
        kps_dict = {}
        with open(lbl_p, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc = float(parts[1]), float(parts[2])
                    kps_dict[cls_id] = (xc, yc)
        
        # Nếu có đủ 14 keypoint (hoặc >= 12)
        if len(kps_dict) >= 12:
            kps_flat = []
            for cid in range(14):
                if cid in kps_dict:
                    kps_flat.extend([kps_dict[cid][0], kps_dict[cid][1]])
                elif (cid + 1) in kps_dict:
                    kps_flat.extend([kps_dict[cid + 1][0], kps_dict[cid + 1][1]])
                else:
                    kps_flat.extend([0.0, 0.0])
            
            json_records.append({
                "id": os.path.relpath(img_p, target_dir).replace("\\", "/"),
                "kps": kps_flat
            })

out_json = os.path.join(target_dir, "data.json")
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(json_records, f, indent=2)

print(f"✅ Đã tạo thành công {out_json} chứa {len(json_records)} ảnh có nhãn 14 keypoints!")
print("=" * 60)
