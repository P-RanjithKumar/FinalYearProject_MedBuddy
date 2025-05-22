# delete_temperature_data.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Vital  # Import your models
from config import Config

# Initialize configuration
config = Config()
engine = create_engine(config.DATABASE_URL)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
db_session = DBSession()

try:
    # Delete all records where vital_type is 'temperature'
    num_deleted = db_session.query(Vital).filter(Vital.vital_type == 'temperature').delete()
    db_session.commit()
    print(f"Deleted {num_deleted} temperature records.")

except Exception as e:
    db_session.rollback()
    print(f"Error deleting temperature data: {e}")

finally:
    db_session.close()