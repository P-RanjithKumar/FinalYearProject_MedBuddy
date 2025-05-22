from sqlalchemy import create_engine

DATABASE_URL = "postgresql://chatbotuser:pass1234@localhost:5432/ragchatbot"

engine = create_engine(DATABASE_URL)

try:
    connection = engine.connect()
    print("Successfully connected to PostgreSQL!")
    connection.close()
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
