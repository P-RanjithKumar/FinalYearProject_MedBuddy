# models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class Chat(Base):
    __tablename__ = 'chats'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(String(500), nullable=True)
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey('chats.id'))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(200))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(Text)
    chunk_index = Column(Integer)
    document = relationship("Document", back_populates="chunks")

class Vital(Base):
    __tablename__ = 'vitals'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    vital_type = Column(String(50))  # e.g., "heart_rate", "temperature"
    value = Column(Integer)

class MedicalHistory(Base):
    __tablename__ = 'medical_history'
    id = Column(Integer, primary_key=True)
    condition = Column(String(200))
    category = Column(String(50))  # e.g., 'surgery', 'allergy', 'chronic'
    diagnosis_date = Column(DateTime)
    notes = Column(Text)

class Prescription(Base):
    __tablename__ = 'prescriptions'
    id = Column(Integer, primary_key=True)
    drug_name = Column(String(150))
    dosage = Column(String(100))
    frequency = Column(String(100))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    purpose = Column(String(200))

class TestResult(Base):
    __tablename__ = 'test_results'
    id = Column(Integer, primary_key=True)
    test_name = Column(String(200))
    result_value = Column(String(100))
    date = Column(DateTime)
    reference_range = Column(String(100))
    lab = Column(String(200))


class Appointment(Base):
    __tablename__ = 'appointments'
    id = Column(Integer, primary_key=True)
    patient_name = Column(String(100), nullable=False)
    appointment_datetime = Column(DateTime, nullable=False)
    doctor_name = Column(String(100), nullable=False)
    appointment_type = Column(String(50))
    status = Column(String(20), default='scheduled')  # scheduled, completed, canceled
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
