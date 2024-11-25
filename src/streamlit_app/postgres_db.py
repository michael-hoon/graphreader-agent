import os
from psycopg import connect
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

DB_URI = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}?sslmode=disable"
)

# Create a connection pool
connection_pool = ConnectionPool(conninfo=DB_URI)

def get_connection():
    return connection_pool.getconn()
