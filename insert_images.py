import os
from PIL import Image
from tqdm import tqdm
from db_connect import get_connection

IMAGE_DIR = r"D:\evoastra major project\images"


def insert_images():
    conn = get_connection()
    if conn is None:
        raise RuntimeError(
            "Database connection failed: get_connection() returned None")

    cur = conn.cursor()
    print(f"✅ Connected to DB. Scanning folder: {IMAGE_DIR}")

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    print(f"🖼️ Found {len(files)} image files.")

    for filename in tqdm(files, desc="Inserting images"):
        img_path = os.path.join(IMAGE_DIR, filename)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            width = height = None

        split = "train"

        cur.execute("""
            INSERT INTO images (image_id, file_path, width, height, split)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO NOTHING;
        """, (filename, img_path, width, height, split))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All images inserted successfully!")


if __name__ == "__main__":
    insert_images()
