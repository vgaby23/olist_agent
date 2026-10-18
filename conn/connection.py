import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

load_dotenv()

def qdrant_connection():
    """Initialize and return a Qdrant client."""
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing in .env")

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Qdrant: {e}")

def sql_connection():
    """Initialize and return a SQLAlchemy engine."""
    DATABASE_URL = os.getenv("DATABASE_URL")

    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to connect to SQL Database: {e}")


# CONNECTION TEST FUNCTION
def test_connections():
    print("--- Testing Database Connections ---")
    
    # Test Qdrant
    qdrant_client = qdrant_connection()
    if qdrant_client:
        try:
            collections = qdrant_client.get_collections()
            print("✅ Qdrant Cloud: Connection Successful!")
            print(f"   Available Collections: {collections.collections}")
        except Exception as e:
            print(f"❌ Qdrant Cloud Error: {e}")
    else:
        print("❌ Qdrant Cloud: Client not initialized.")

    # Test SQL
    sql_engine = sql_connection()
    if sql_engine:
        try:
            with sql_engine.connect() as conn:
                # Run a simple query to ensure the DB is responsive
                result = conn.execute(text("SELECT sqlite_version();")).fetchone()
                print("✅ SQL Database: Connection Successful!")
                print(f"   SQLite Version: {result[0]}")
        except Exception as e:
            print(f"❌ SQL Database Error: {e}")
    else:
        print("❌ SQL Database: Engine not initialized.")
    
    print("------------------------------------")

if __name__ == "__main__":
    test_connections()