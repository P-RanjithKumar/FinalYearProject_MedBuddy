from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Vital  # Import the Vital model
from config import Config

# Initialize configuration (same as in app2.py)
config = Config()
engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)

db = DBSession()

# Generate sample data for the past 4 weeks (28 days)
today = datetime.now()
for i in range(28):
    date = today - timedelta(days=i)
    vital_types = ["heart_rate", "temperature", "blood_pressure"]
    vital_type = random.choice(vital_types)
    if vital_type == "heart_rate":
        value = random.randint(60, 100)  # Normal heart rate range
    elif vital_type == "temperature":
        value = random.randint(97, 99)  # Normal temperature range (Fahrenheit)
    else:  # blood_pressure
        value = random.randint(90,140)

    new_vital = Vital(date=date, vital_type=vital_type, value=value)
    db.add(new_vital)

db.commit()
db.close()

print("Sample vital data added.")