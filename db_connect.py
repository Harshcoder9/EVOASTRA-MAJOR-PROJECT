import psycopg2

def get_connection():
    try:
        conn = psycopg2.connect(
            dbname="flickr8k_db",
            user="postgres",
            password="1234",   # replace with your actual password
            host="localhost",
            port="5432"
        )
        print("✅ Database connected successfully!")
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        return None
if __name__ == "__main__":
    get_connection()

