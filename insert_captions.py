from db_connect import get_connection
from tqdm import tqdm

CAPTIONS_FILE = "../Flickr8k.token.txt"

def insert_captions():
    conn = get_connection()
    if conn is None:
        raise RuntimeError("Database connection failed: get_connection() returned None")
    cur = conn.cursor()

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            try:
                image_part, caption = line.split("\t")
            except ValueError:
                parts = line.split(" ", 1)
                image_part = parts[0]
                caption = parts[1] if len(parts) > 1 else ""

            image_id = image_part.split("#")[0]

            cur.execute("""
                INSERT INTO captions (image_id, caption_text)
                VALUES (%s, %s)
            """, (image_id, caption))
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ All captions inserted successfully!")

if __name__ == "__main__":
    insert_captions()
