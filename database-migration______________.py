from sqlalchemy import text
from MedBuddyApp import engine  # Import your SQLAlchemy engine

def run_migrations():
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM medical_history"))
        conn.execute(text("DELETE FROM prescriptions"))
        conn.execute(text("DELETE FROM test_results"))
        conn.commit()

if __name__ == '__main__':
    run_migrations()

