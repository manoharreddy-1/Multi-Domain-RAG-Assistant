import os
from pymongo import MongoClient

# Use the provided MongoDB Atlas URI or fallback to environment variable
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://bommareddymanoharreddy999_db_user:B.manohar%401@cluster0.zzgruyd.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = 'rag_db'

# Global DB connection (lazy loaded)
_db = None
_client = None

def get_db():
    global _db, _client
    if _db is None:
        primary_uri = os.getenv('MONGODB_URI', 'mongodb+srv://bommareddymanoharreddy999_db_user:B.manohar%401@cluster0.zzgruyd.mongodb.net/?retryWrites=true&w=majority')
        try:
            print(f"Attempting to connect to primary MongoDB...")
            client = MongoClient(primary_uri, serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            _client = client
            _db = _client[DB_NAME]
            print("Connected to primary MongoDB cluster successfully.")
        except Exception as e:
            print(f"Primary MongoDB connection failed: {e}. Falling back to local MongoDB...")
            local_uri = 'mongodb://localhost:27017/'
            try:
                client = MongoClient(local_uri, serverSelectionTimeoutMS=2000)
                client.admin.command('ping')
                _client = client
                _db = _client[DB_NAME]
                print("Connected to local fallback MongoDB successfully.")
            except Exception as local_err:
                print(f"Local fallback MongoDB connection also failed: {local_err}")
                raise e
    return _db

def get_db_client():
    global _client
    if _client is None:
        get_db()
    return _client

def init_db():
    try:
        db = get_db()
        # Ensure collections exist simply by pinging or inserting/deleting a dummy doc if needed
        # MongoDB creates collections lazily on first insert.
        # But we can define indexes here.
        db.users.create_index("email", unique=True)
        print("MongoDB Initialized successfully. Connected to cluster using db:", DB_NAME)
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")

if __name__ == '__main__':
    print("Initializing MongoDB...")
    init_db()
