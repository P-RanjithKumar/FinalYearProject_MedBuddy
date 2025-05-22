from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Base, Chat, Message, Document, DocumentChunk, MedicalHistory, Prescription, TestResult, Vital, Appointment
from rag import RAGManager
from ocr_processor import PrescriptionOCR
from config import Config
import requests
import json
from datetime import datetime
import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import pyttsx3
import logging
import uuid
import time
import re
import whisper

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this

# Initialize configuration
config = Config()
engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)

# Initialize RAG manager
rag_manager = RAGManager(config)

# Check if API key exists before initializing
if config.GEMINI_API_KEY:
    try:
        ocr_processor = PrescriptionOCR(api_key=config.GEMINI_API_KEY)
        logging.info("PrescriptionOCR (Gemini) initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize PrescriptionOCR (Gemini): {e}", exc_info=True)
        ocr_processor = None # Set to None if initialization fails
else:
    logging.warning("GEMINI_API_KEY not found in config. Prescription OCR upload will not work.")
    ocr_processor = None

# Initialize text-to-speech engine
engine_tts = pyttsx3.init()

# --- Whisper Initialization ---
WHISPER_MODEL_NAME = "base.en" # Using the model specified in testeroneeeeee.py
whisper_model = None
try:
    logging.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
    # Load the model. You might want to specify device='cuda' if you have a GPU
    # For CPU or potential CUDA issues, using fp16=False is safer.
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model '{WHISPER_MODEL_NAME}': {e}", exc_info=True)
    # Depending on severity, you might want to exit or raise a critical error
    raise SystemExit(f"Critical Error: Could not load Whisper model '{WHISPER_MODEL_NAME}'.")


# Temporary directory for audio files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Ensure ffmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info("FFmpeg found.")
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please install FFmpeg and add it to PATH.")
        raise Exception("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg version check failed: {e.stderr.decode()}")
        raise Exception(f"FFmpeg check failed: {e.stderr.decode()}")


check_ffmpeg()

# Convert WebM to WAV using ffmpeg (Keep this as Whisper works well with WAV)
def convert_webm_to_wav(input_path):
    """Converts WebM audio file to WAV format (16kHz, mono, PCM s16le)."""
    output_path = os.path.splitext(input_path)[0] + '.wav'
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le', # Standard WAV codec
            '-ar', '16000',         # Sample rate Whisper handles well
            '-ac', '1',             # Mono channel
            '-y',                   # Overwrite output file if it exists
            output_path
        ]
        logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30) # Added timeout
        logging.info(f"FFmpeg conversion successful: {input_path} -> {output_path}")
        return output_path
    except subprocess.TimeoutExpired:
        logging.error(f"FFmpeg conversion timed out for {input_path}")
        raise Exception("FFmpeg conversion timed out.")
    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg conversion failed for {input_path}. Error: {e.stderr.decode()}"
        logging.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        logging.error(f"Unexpected error during FFmpeg conversion: {str(e)}", exc_info=True)
        raise Exception(f"FFmpeg conversion failed: {str(e)}")

# Speech-to-Text (STT) endpoint using Whisper
@app.route('/stt', methods=['POST'])
def speech_to_text():
    """Convert speech audio to text using Whisper"""
    if whisper_model is None:
         return jsonify({'error': 'Speech recognition service not available (model not loaded)'}), 503

    try:
        # Validate request
        if 'audio' not in request.files:
            logging.warning("STT request missing 'audio' file part.")
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if not audio_file or audio_file.filename == '':
            logging.warning("STT request received empty audio file.")
            return jsonify({'error': 'Empty or no filename'}), 400

        # Secure filename and prepare paths
        original_filename = secure_filename(audio_file.filename)
        file_prefix = f"stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        temp_webm_path = os.path.join(TEMP_DIR, f"{file_prefix}.webm")
        wav_path = None
        start_time = time.time()

        try:
            # Save uploaded WebM file
            logging.info(f"Saving uploaded audio to: {temp_webm_path}")
            audio_file.save(temp_webm_path)

            # Convert to WAV format
            logging.info(f"Converting {temp_webm_path} to WAV...")
            wav_path = convert_webm_to_wav(temp_webm_path)

            # --- Transcribe using Whisper ---
            logging.info(f"Starting Whisper transcription for: {wav_path}")
            # Use fp16=False if running on CPU or having CUDA issues
            # Set language='en' if you are sure it's English, otherwise Whisper detects it
            transcription_options = {"fp16": False, "language": "en"}
            result = whisper_model.transcribe(wav_path, **transcription_options)
            full_text = result["text"].strip()
            logging.info(f"Whisper transcription successful. Result: '{full_text}'")
            # --- End Whisper Transcription ---

            processing_time = time.time() - start_time
            logging.info(f"STT processing time: {processing_time:.2f} seconds")

            return jsonify({
                'text': full_text,
                'processing_time': processing_time
            })

        finally:
            # Cleanup temporary files
            for path in [temp_webm_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Cleaned up temporary file: {path}")
                    except Exception as e:
                        logging.warning(f"File cleanup error for {path}: {str(e)}")

    except Exception as e:
        # Log the full error with traceback
        logging.error(f"STT Error: {str(e)}", exc_info=True)
        # Provide a generic error message to the client
        return jsonify({
            'error': 'Speech recognition failed',
            'details': 'An internal server error occurred during transcription.' # Avoid exposing internal details
        }), 500


# Text-to-Speech (TTS) endpoint (No changes needed here)
@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate unique filename
        unique_id = str(uuid.uuid4())
        # Ensure static/tts directory exists
        tts_dir = Path('static/tts')
        tts_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(tts_dir / f'{unique_id}.mp3') # Use Path object for robustness

        # Configure engine properties
        engine_tts.setProperty('rate', 150)  # Adjust speech rate
        engine_tts.setProperty('volume', 0.9)  # Adjust volume

        # Save to unique file
        logging.info(f"Generating TTS audio for text: '{text[:50]}...' to {output_path}")
        engine_tts.save_to_file(text, output_path)
        engine_tts.runAndWait() # This blocks until saving is complete

        # Check if file was created
        if not os.path.exists(output_path):
             logging.error(f"TTS file was not created: {output_path}")
             raise IOError("TTS audio file generation failed.")

        logging.info(f"TTS audio generated successfully: {output_path}")
        return jsonify({
            'audio_url': f'/static/tts/{unique_id}.mp3',
            'expires_at': datetime.now().timestamp() + 3600  # 1 hour expiration (consider a proper cleanup mechanism)
        })

    except Exception as e:
        logging.error(f"TTS Error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Text-to-speech generation failed'}), 500

# main system functionalities :
@app.route('/')
def home():
    db = DBSession()
    chats = db.query(Chat).order_by(Chat.created_at.desc()).all()
    db.close()
    return render_template('index.html', chats=chats)

@app.route('/chat/<int:chat_id>')
def get_chat(chat_id):
    db = DBSession()
    chat = db.get(Chat, chat_id)  # Updated to SQLAlchemy 2.0 syntax
    messages = []
    if chat:
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            for msg in db.query(Message)
                     .filter(Message.chat_id == chat_id)
                     .order_by(Message.timestamp.asc())
                     .all()
        ]
    db.close()
    return jsonify({"messages": messages})

@app.route('/chat/new', methods=['POST'])
def new_chat():
    db = DBSession()
    chat = Chat(title="Chat",created_at= datetime.now(),summary="The description of this chat")
    db.add(chat)
    db.commit()
    print(chat.id)
    chat_id = chat.id
    db.close()
    return jsonify({"chat_id": chat_id})

@app.route('/chat/<int:chat_id>/message', methods=['POST'])
@stream_with_context
def send_message(chat_id):
    def generate():
        db = DBSession()
        try:
            user_message = request.json.get('message', '').strip()
            if not user_message:
                yield json.dumps({'error': 'Empty message'})
                return

            chat = db.get(Chat, chat_id)  # Updated to SQLAlchemy 2.0 syntax
            if not chat:
                yield json.dumps({'error': 'Chat not found'})
                return

            # Get chat history
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in chat.messages[-5:]  # Last 5 messages for context
            ]
            # Get relevant context from RAG
            context = rag_manager.get_relevant_chunks(user_message, chat_history)

            user_msg = Message(chat_id=chat_id, role="user", content=user_message)
            db.add(user_msg)
            db.commit()

            # Prepare messages for Ollama API with improved system prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical assistant. If provided with context, only use it "
                        "when it's directly relevant to the user's question. If the context "
                        "isn't relevant to the current question, ignore it completely. "
                        f"\n\nAvailable context:\n{context}" if context else ""
                    )
                }
            ] + chat_history + [
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            response = requests.post(config.OLLAMA_API_URL, json={"model": "llama3.2:1b", "messages": messages}, stream=True)
            response.raise_for_status()

            assistant_message = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if json_response.get('done', False):
                        # Send final token with end-of-stream marker
                        yield json.dumps({
                            'token': chunk,
                            'is_final': True  # Add final token marker
                        })
                        break
                        
                    chunk = json_response.get('message', {}).get('content', '')
                    assistant_message += chunk
                    yield json.dumps({'token': chunk})  # Stream the chunk

            # Persist the complete assistant message to the database
            assistant_msg = Message(chat_id=chat_id, role="assistant", content=assistant_message)
            db.add(assistant_msg)
            db.commit()

        except Exception as e:
            yield json.dumps({
                'error': f'Server error: {str(e)}',
                'is_final': True  # Ensure stream closure on error
            })
        finally:
            db.close()

    return Response(generate(), mimetype='application/json')

"""
@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    db = None

    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_TEXT, filename)
        
        # Ensure the upload folder exists
        os.makedirs(config.UPLOAD_TEXT, exist_ok=True)

        # Save the file to disk
        file.seek(0)  # Reset file cursor position
        file.save(filepath)
        
        # Detect file type (text vs. binary)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            content = "[BINARY FILE: Cannot display content]"
        
        # Save metadata and content to database
        db = DBSession()
        document = Document(filename=filename, content=content)
        db.add(document)
        db.commit()
        
        # Add to vector store
        num_chunks = rag_manager.add_document(
            str(document.id),
            content,
            {"filename": file.filename}
        )
        
        return jsonify({
            'message': f'Successfully processed document into {num_chunks} chunks'
        })
    
    except Exception as e:
        logging.exception("Error in /upload route:")  # Log the full traceback
        return jsonify({'error': str(e)}), 500

    finally:
        if db:
            db.close()
"""

# Add to app2.py
def analyze_document(content):
    """Send document to LLM for medical information extraction"""
    prompt = f"""Analyze this medical document and extract structured information:
    {content}
    
    Return JSON format with:
    - medical_history: list of conditions/surgeries/allergies with dates
    - prescriptions: list of medications with dosage, frequency
    - test_results: list of tests with values, dates, reference ranges
    Example format:
    {{
        "medical_history": [
            {{"condition": "Diabetes", "category": "chronic", "diagnosis_date": "2015-03-15"}}
        ],
        "prescriptions": [
            {{"drug_name": "Metformin", "dosage": "500mg", "frequency": "twice daily"}}
        ],
        "test_results": [
            {{"test_name": "HbA1c", "result_value": "6.2%", "date": "2024-01-15", "reference_range": "4-5.6%"}}
        ]
    }}
    Return only valid JSON, no markdown."""
    
    try:
        response = requests.post(
            config.OLLAMA_API_URL,
            json={
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json"
            },
            timeout=1000
        )
        response.raise_for_status()
        return response.json().get('message', {}).get('content', '{}')
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    db = None

    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_TEXT, filename)
        
        # Ensure the upload folder exists
        os.makedirs(config.UPLOAD_TEXT, exist_ok=True)

        # Save the file to disk
        file.seek(0)  # Reset file cursor position
        file.save(filepath)
        
        # Detect file type (text vs. binary)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            content = "[BINARY FILE: Cannot display content]"
        
        # Save metadata and content to database
        db = DBSession()
        document = Document(filename=filename, content=content)
        db.add(document)
        db.commit()
        
        # Add to vector store
        num_chunks = rag_manager.add_document(
            str(document.id),
            content,
            {"filename": file.filename}
        )

        analysis_result = analyze_document(content)

        if analysis_result:
            try:
                data = json.loads(analysis_result)
                db = DBSession()
                
                # Store medical history
                for item in data.get('medical_history', []):
                    db.add(MedicalHistory(
                        condition=item.get('condition'),
                        category=item.get('category'),
                        diagnosis_date=datetime.strptime(item.get('diagnosis_date'), '%Y-%m-%d') if item.get('diagnosis_date') else None,
                        notes=item.get('notes')
                    ))
                
                # Store prescriptions
                for item in data.get('prescriptions', []):
                    db.add(Prescription(
                        drug_name=item.get('drug_name'),
                        dosage=item.get('dosage'),
                        frequency=item.get('frequency'),
                        start_date=datetime.strptime(item.get('start_date'), '%Y-%m-%d') if item.get('start_date') else None,
                        purpose=item.get('purpose')
                    ))
                
                # Store test results
                for item in data.get('test_results', []):
                    db.add(TestResult(
                        test_name=item.get('test_name'),
                        result_value=item.get('result_value'),
                        date=datetime.strptime(item.get('date'), '%Y-%m-%d') if item.get('date') else None,
                        reference_range=item.get('reference_range'),
                        lab=item.get('lab')
                    ))
                
                db.commit()
                analysis_success = True
            except Exception as e:
                db.rollback()
                logging.error(f"Error saving analysis: {str(e)}")
            finally:
                if db:
                    db.close()
        return jsonify({
            'message': f'Document processed successfully with {num_chunks} chunks',
            'analysis_performed': analysis_success
        }), 200

    except Exception as e:
        logging.exception("Error in /upload route:")
        return jsonify({'error': str(e)}), 500
    finally:
        if db:
            db.close()

@app.route('/api/medical-history')
def get_medical_history():
    db = DBSession()
    history = db.query(MedicalHistory).all()
    return jsonify([{
        'condition': h.condition,
        'category': h.category,
        'diagnosis_date': h.diagnosis_date.strftime('%Y-%m-%d') if h.diagnosis_date else None,
        'notes': h.notes
    } for h in history])

@app.route('/api/prescriptions')
def get_prescriptions():
    db = DBSession()
    scripts = db.query(Prescription).all()
    return jsonify([{
        'drug_name': p.drug_name,
        'dosage': p.dosage,
        'frequency': p.frequency,
        'start_date': p.start_date.strftime('%Y-%m-%d') if p.start_date else None,
        'purpose': p.purpose
    } for p in scripts])

@app.route('/api/test-results')
def get_test_results():
    db = DBSession()
    tests = db.query(TestResult).all()
    return jsonify([{
        'test_name': t.test_name,
        'result_value': t.result_value,
        'date': t.date.strftime('%Y-%m-%d') if t.date else None,
        'reference_range': t.reference_range,
        'lab': t.lab
    } for t in tests])


############    image prescription    #################

@app.route('/upload/prescription', methods=['POST'])
def upload_prescription():
    # --- Check if OCR processor is available ---
    if ocr_processor is None:
         logging.error("Prescription upload endpoint called, but OCR processor (Gemini) is not initialized.")
         return jsonify({'error': 'Prescription processing service is unavailable. Check API key.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Basic check for likely image types (can be more robust)
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'}), 415

    db = None
    temp_image_path = None
    analysis_success = False # Flag for analysis status

    try:
        # --- Save Uploaded Image Temporarily ---
        original_filename = secure_filename(file.filename)
        # Create a unique temporary filename for the image
        temp_suffix = Path(original_filename).suffix
        temp_image_fd, temp_image_path = tempfile.mkstemp(suffix=temp_suffix, dir=config.UPLOAD_IMAGES)
        os.close(temp_image_fd) # Close the file descriptor

        file.seek(0)
        file.save(temp_image_path)
        logging.info(f"Temporarily saved uploaded prescription image to: {temp_image_path}")

        # --- Perform OCR using Gemini ---
        logging.info(f"Starting Gemini OCR for image: {original_filename}")
        extracted_text = ocr_processor.image_to_text(temp_image_path)
        # Handle potential blocked content or errors from OCR
        if extracted_text is None or "[Content Blocked" in extracted_text or "[Prompt Blocked" in extracted_text:
             logging.warning(f"OCR for {original_filename} resulted in blocked or empty content: {extracted_text}")
             # Decide how to proceed: maybe save a document with placeholder text, or return error
             # For now, let's return an error indicating OCR issue
             raise ValueError(f"OCR failed or content blocked for {original_filename}. Text: {extracted_text}")

        # --- Process Extracted Text Like a Text Document ---
        # Now, follow the logic similar to the '/upload' route for text files

        # 1. Save Document to Database
        db = DBSession()
        document = Document(
            filename=original_filename, # Use original filename
            content=extracted_text       # Store the OCR'd text
        )
        db.add(document)
        db.commit() # Commit to get the ID
        doc_id = document.id
        logging.info(f"Saved OCR text from '{original_filename}' to DB Document ID: {doc_id}")

        # 2. Add to Vector Store (RAG)
        num_chunks = 0
        if extracted_text:
            try:
                num_chunks = rag_manager.add_document(
                    str(doc_id),
                    extracted_text,
                    {"filename": original_filename, "type": "prescription_image_ocr"} # Specific metadata
                )
                logging.info(f"Added OCR text (Doc ID {doc_id}) to RAG with {num_chunks} chunks.")
            except Exception as e:
                logging.error(f"Failed to add OCR text (Doc ID {doc_id}) to RAG: {e}", exc_info=True)
                # Consider if this failure should prevent further steps
        else:
            logging.warning(f"Skipping RAG indexing for empty OCR text (Doc ID {doc_id})")


        # 3. Analyze the Extracted Text (using the same 'analyze_document' function)
        analysis_result_json = None
        if extracted_text:
            logging.info(f"Starting analysis for OCR text (Doc ID {doc_id})...")
            analysis_result_json = analyze_document(extracted_text)
        else:
             logging.info(f"Skipping analysis for empty OCR text (Doc ID {doc_id})")


        # 4. Save Structured Analysis Results to DB
        if analysis_result_json:
            try:
                # Ensure session is active
                if not db.is_active: db = DBSession()

                data = json.loads(analysis_result_json) # Parse JSON string
                logging.info(f"Analysis of OCR text successful (Doc ID {doc_id}). Saving structured data.")

                # Store medical history (extracted from OCR'd text)
                for item in data.get('medical_history', []):
                    db.add(MedicalHistory(
                        condition=item.get('condition'),
                        category=item.get('category'),
                        diagnosis_date=datetime.strptime(item.get('diagnosis_date'), '%Y-%m-%d') if item.get('diagnosis_date') else None,
                        notes=item.get('notes')
                    ))

                # Store prescriptions (extracted from OCR'd text - goes into the 'Prescription' table)
                for item in data.get('prescriptions', []):
                    db.add(Prescription(
                        drug_name=item.get('drug_name'),
                        dosage=item.get('dosage'),
                        frequency=item.get('frequency'),
                        start_date=datetime.strptime(item.get('start_date'), '%Y-%m-%d') if item.get('start_date') else None,
                        purpose=item.get('purpose')
                    ))

                # Store test results (extracted from OCR'd text)
                for item in data.get('test_results', []):
                    db.add(TestResult(
                        test_name=item.get('test_name'),
                        result_value=item.get('result_value'),
                        date=datetime.strptime(item.get('date'), '%Y-%m-%d') if item.get('date') else None,
                        reference_range=item.get('reference_range'),
                        lab=item.get('lab')
                    ))

                db.commit()
                analysis_success = True
                logging.info(f"Successfully saved structured analysis data from OCR text (Doc ID {doc_id}).")

            except json.JSONDecodeError as e:
                 logging.error(f"Failed to parse analysis JSON for OCR text (Doc ID {doc_id}): {e}. JSON string: {analysis_result_json}")
                 db.rollback()
            except Exception as e:
                db.rollback()
                logging.error(f"Error saving analysis results from OCR text (Doc ID {doc_id}): {str(e)}", exc_info=True)
        else:
             logging.info(f"No analysis performed or analysis failed for OCR text (Doc ID {doc_id}).")

        # --- Success Response ---
        return jsonify({
            'message': f'Successfully processed prescription image "{original_filename}"' + (f' into {num_chunks} chunks.' if num_chunks > 0 else '.'),
            'document_id': doc_id,
            'extracted_text_preview': extracted_text[:200] + ('...' if len(extracted_text) > 200 else ''), # Preview
            'analysis_performed': analysis_success
        }), 200

    except FileNotFoundError as e:
         logging.error(f"File not found error during prescription upload: {e}", exc_info=True)
         return jsonify({'error': f'File system error: {str(e)}'}), 500
    except ValueError as e: # Catch specific errors like OCR failure/blocking
        logging.warning(f"Value error during prescription upload for {original_filename}: {e}")
        if db and db.is_active: db.rollback()
        return jsonify({'error': str(e)}), 400 # Return bad request for OCR issues
    except RuntimeError as e: # Catch runtime errors from OCR process
         logging.error(f"Runtime error during prescription upload (likely Gemini issue): {e}", exc_info=True)
         if db and db.is_active: db.rollback()
         return jsonify({'error': f'Failed to process prescription image: {str(e)}'}), 500
    except Exception as e:
        logging.error(f"Unexpected error in /upload/prescription for {original_filename}: {str(e)}", exc_info=True)
        if db and db.is_active:
            db.rollback()
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
    finally:
        # --- Clean up temporary image file ---
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logging.info(f"Cleaned up temporary image file: {temp_image_path}")
            except Exception as e:
                logging.warning(f"Error cleaning up temp image file {temp_image_path}: {str(e)}")
        # --- Close DB Session ---
        if db and db.is_active:
            db.close()

@app.route('/chat/<int:chat_id>/delete', methods=['DELETE'])
def delete_chat(chat_id):
    try:
        db = DBSession()
        chat = db.get(Chat, chat_id)  # Updated to SQLAlchemy 2.0 syntax
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete the chat (cascade will handle related messages)
        db.delete(chat)
        db.commit()
        
        return jsonify({'message': 'Chat deleted successfully'})
    
    except Exception as e:
        db.rollback()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        db.close()


@app.route('/documents', methods=['GET'])
def get_documents():
    try:
        db = DBSession()
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        documents_list = [
            {
                'id': doc.id,
                'filename': doc.filename,
                'created_at': doc.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'prescription' if doc.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')) else 'text'
            }
            for doc in documents
        ]
        return jsonify({'documents': documents_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/documents/<int:doc_id>/view', methods=['GET'])
def view_text_document(doc_id):
    db = DBSession()
    document = db.query(Document).get(doc_id)
    db.close()
    if not document:
        return "Document not found", 404
    # Render a dedicated template that shows the document content in an overlay
    return render_template('text_viewer.html', document=document)

@app.route('/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        db = DBSession()
        document = db.query(Document).get(doc_id)
        
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete from database
        db.delete(document)
        
        # Delete from vector store
        collection = rag_manager.collection
        collection.delete(
            where={"doc_id": str(doc_id)}
        )
        
        db.commit()
        return jsonify({'message': 'Document deleted successfully'})
    
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

# Add this new route to app.py
@app.route('/chat/<int:chat_id>/title', methods=['PUT'])
def update_chat_title(chat_id):
    try:
        new_title = request.json.get('title', '').strip()
        
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400

        db = DBSession()
        chat = db.get(Chat, chat_id)  # Updated to SQLAlchemy 2.0 syntax
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404

        chat.title = new_title
        db.commit()
        
        return jsonify({
            'message': 'Title updated successfully',
            'title': new_title
        })
    
    except Exception as e:
        db.rollback()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        db.close()

@app.route('/chat/<int:chat_id>/summary', methods=['POST'])
def generate_chat_summary(chat_id):
    try:
        db = DBSession()
        chat = db.get(Chat, chat_id)  # Updated to SQLAlchemy 2.0 syntax
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404

        # Get last 6 messages from database
        messages = [f"{msg.role}: {msg.content}" for msg in chat.messages[-6:]]
        if not messages:
            return jsonify({'error': 'No messages to summarize'}), 400

        # Improved medical summary prompt
        prompt = f"""Create a concise 25-word medical summary .        Conversation excerpts:{" | ".join(messages)}"""

        # Make request to Ollama with streaming disabled
        response = requests.post(
            config.OLLAMA_API_URL,
            json={
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # Disable streaming
            },
            timeout=1000
        )

        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response with proper error handling
        try:
            full_response = response.json()
            summary = full_response['message']['content'].strip()
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Ollama response parsing failed: {response.text}")
            raise Exception("Failed to parse LLM response") from e

        # Clean and truncate summary
        summary = re.sub(r'\s+', ' ', summary)[:250]  # Remove extra spaces and truncate
        
        # Update database
        chat.summary = summary
        db.commit()
        
        return jsonify({"summary": summary})

    except requests.exceptions.RequestException as e:
        error_msg = f"LLM API Connection Error: {str(e)}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 502
    except Exception as e:
        logging.error(f"Summary Error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Summary generation failed: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/chat/<int:chat_id>/generate-title', methods=['POST'])
def generate_chat_title(chat_id):
    try:
        db = DBSession()
        chat = db.get(Chat, chat_id)
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404

        messages = [f"{msg.role}: {msg.content}" for msg in chat.messages[:6]]
        if not messages:
            return jsonify({'error': 'No messages to generate title'}), 400

        prompt = """Create a concise title (5 words) for this medical conversation.
        Focus on: main condition, core symptoms, and key advice.
        Use simple language. Example: 'Hypertension: Blood Pressure and Lifestyle Tips'
        Remember: Strictly 5 words only."""

        response = requests.post(
            config.OLLAMA_API_URL,
            json={
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=1000
        )
        response.raise_for_status()

        generated_title = response.json()['message']['content'].strip()
        generated_title = re.sub(r'\s+', ' ', generated_title)[:50].strip()

        chat.title = generated_title
        db.commit()

        return jsonify({"title": generated_title})

    except Exception as e:
        db.rollback()
        logging.error(f"Title generation error: {str(e)}")
        return jsonify({'error': 'Title generation failed'}), 500
    finally:
        db.close()






@app.route('/vitals', methods=['POST'])
def add_vital():
    try:
        data = request.get_json()
        date_str = data.get('date')
        vital_type = data.get('vital_type')
        value = data.get('value')

        if not all([date_str, vital_type, value]):
            return jsonify({'error': 'Missing data'}), 400

        date = datetime.strptime(date_str, '%Y-%m-%d')

        db = DBSession()
        new_vital = Vital(date=date, vital_type=vital_type, value=value)
        db.add(new_vital)
        db.commit()
        return jsonify({'message': 'Vital sign added successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'db' in locals():
            db.close()


@app.route('/vitals/trends', methods=['GET'])
def get_vital_trends():
    try:
        db = DBSession()
        vitals = db.query(Vital).order_by(Vital.date.desc()).all()
        vitals_list = [
            {
                'id': vital.id,
                'date': vital.date.strftime('%Y-%m-%d'),
                'vital_type': vital.vital_type,
                'value': vital.value
            }
            for vital in vitals
        ]
        return jsonify({'vitals': vitals_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()
@app.route('/vitals/<int:vital_id>', methods=['DELETE'])
def delete_vital(vital_id):
    db = DBSession()
    try:
        success = delete_vital_data(db, vital_id)
        if success:
            return jsonify({'message': 'Vital data deleted successfully'}), 200
        else:
            return jsonify({'error': 'Vital data not found'}), 404
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

def delete_vital_data(db_session, vital_id):
    """Deletes a vital data entry by its ID.

    Args:
        db_session: The database session object.
        vital_id: The ID of the vital data entry to delete.

    Returns:
        True if the vital data was deleted successfully, False otherwise.
    """
    try:
        vital = db_session.query(Vital).get(vital_id)
        if vital:
            db_session.delete(vital)
            db_session.commit()
            return True
        else:
            return False  # Vital data not found
    except Exception as e:
        print(f"Error deleting vital data: {e}")  # Consider logging this
        return False

    
# Appointment management routes
@app.route('/appointments', methods=['POST'])
def create_appointment():
    data = request.get_json()
    db = DBSession()
    try:
        appointment = Appointment(
            patient_name=data['patient_name'],
            appointment_datetime=datetime.fromisoformat(data['appointment_datetime']),
            doctor_name=data['doctor_name'],
            appointment_type=data.get('type', 'general'),
            notes=data.get('notes', '')
        )
        db.add(appointment)
        db.commit()
        return jsonify({
            'id': appointment.id,
            'message': 'Appointment created successfully'
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        db.close()

@app.route('/appointments')
def get_appointments():
    db = DBSession()
    appointments = db.query(Appointment).order_by(Appointment.appointment_datetime).all()
    now = datetime.utcnow()
    
    def categorize(appt):
        appt_time = appt.appointment_datetime
        if appt_time.date() < now.date():
            return 'past'
        elif appt_time.date() == now.date():
            return 'today'
        else:
            return 'upcoming'
    
    categorized = {
        'past': [],
        'today': [],
        'upcoming': []
    }
    
    for appt in appointments:
        appt_data = {
            'id': appt.id,
            'patient_name': appt.patient_name,
            'datetime': appt.appointment_datetime.isoformat(),
            'doctor': appt.doctor_name,
            'type': appt.appointment_type,
            'status': appt.status,
            'notes': appt.notes
        }
        categorized[categorize(appt)].append(appt_data)
    
    db.close()
    return jsonify(categorized)

@app.route('/appointments/<int:appointment_id>', methods=['PUT'])
def update_appointment(appointment_id):
    data = request.get_json()
    db = DBSession()
    try:
        appointment = db.get(Appointment, appointment_id)
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404
            
        if 'status' in data:
            appointment.status = data['status']
        if 'notes' in data:
            appointment.notes = data['notes']
        if 'appointment_datetime' in data:
            appointment.appointment_datetime = datetime.fromisoformat(data['appointment_datetime'])
            
        db.commit()
        return jsonify({'message': 'Appointment updated successfully'})
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        db.close()

@app.route('/appointments/<int:appointment_id>', methods=['GET'])
def get_appointment_details(appointment_id):
    db = DBSession()
    try:
        # Use db.get() for efficient primary key lookup in SQLAlchemy 2.0+
        appointment = db.get(Appointment, appointment_id)

        if not appointment:
            # If no appointment found with that ID, return 404
            logging.warning(f"GET /appointments/{appointment_id}: Appointment not found.")
            return jsonify({'error': 'Appointment not found'}), 404

        # Serialize the appointment data into a dictionary
        appt_data = {
            'id': appointment.id,
            'patient_name': appointment.patient_name,
            'datetime': appointment.appointment_datetime.isoformat(), # Use ISO format for JS compatibility
            'doctor': appointment.doctor_name,
            'type': appointment.appointment_type,
            'status': appointment.status,
            'notes': appointment.notes
        }
        # Return the data as JSON with a 200 OK status
        return jsonify(appt_data), 200

    except Exception as e:
        # Log the error for debugging
        logging.error(f"Error fetching appointment details for ID {appointment_id}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error fetching appointment details'}), 500
    finally:
        # Ensure the database session is closed
        if db:
            db.close()

if __name__ == '__main__':
    app.run(debug=True)