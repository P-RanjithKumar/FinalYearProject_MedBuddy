# ocr_processor.py
import google.generativeai as genai
from PIL import Image
import logging
from datetime import datetime
from pathlib import Path # Use pathlib

class PrescriptionOCR:
    def __init__(self, api_key: str):
        """
        Initializes the OCR processor with the Google Gemini API Key.

        Args:
            api_key (str): The Google AI API Key.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if not api_key:
            self.logger.error("Google Gemini API Key is required but not provided.")
            raise ValueError("API key missing for PrescriptionOCR.")

        try:
            genai.configure(api_key=api_key)
            # Using Gemini 1.5 Flash - often good balance of speed/capability
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.logger.info("Gemini 1.5 Flash model configured successfully.")
        except Exception as e:
            self.logger.error(f"Failed to configure Google Gemini: {str(e)}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Gemini model: {e}")

    def image_to_text(self, image_path: str) -> str:
        """
        Extracts text from a prescription image using Google Gemini.

        Args:
            image_path (str): Path to the prescription image file.

        Returns:
            str: Extracted raw text from the image.

        Raises:
            FileNotFoundError: If the image path does not exist.
            Exception: For API errors or other processing issues.
        """
        image_path_obj = Path(image_path)
        if not image_path_obj.is_file():
            self.logger.error(f"Image file not found at: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Processing prescription image: {image_path}")

        try:
            img = Image.open(image_path)

            # --- Gemini API Call ---
            # Simple prompt focusing on accurate transcription
            prompt = """Extract all text content from this image of a medical prescription.
Present the text exactly as it appears, maintaining the structure and line breaks as accurately as possible.
Do not add any explanations, summaries, or interpretations. Only return the raw text found in the image."""

            # Generate content
            response = self.model.generate_content([prompt, img], stream=False) # Use stream=False for single response

            # --- Response Handling ---
            # Check for safety ratings or blocks if necessary (optional but good practice)
            if not response.candidates:
                 # Handle cases where the model didn't generate a candidate (e.g., safety blocked)
                 safety_feedback = response.prompt_feedback
                 block_reason = safety_feedback.block_reason if safety_feedback else "Unknown"
                 self.logger.warning(f"Gemini blocked the response for {image_path}. Reason: {block_reason}")
                 # Consider how to handle blocked content - maybe return empty string or raise specific error
                 return f"[Content Blocked by Safety Filter: {block_reason}]" # Or return ""

            # Extract text - accessing parts safely
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                extracted_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                self.logger.warning(f"Could not extract text from Gemini response for {image_path}. Response: {response}")
                extracted_text = "" # Default to empty if no text found

            cleaned_text = extracted_text.strip()
            self.logger.info(f"Successfully extracted text from {image_path}. Length: {len(cleaned_text)}")
            # print(f"--- Extracted Text Start ---\n{cleaned_text}\n--- Extracted Text End ---") # Optional: for debugging
            return cleaned_text

        except FileNotFoundError: # Already checked, but belts and suspenders
            raise
        except genai.types.generation_types.BlockedPromptException as bpe:
             self.logger.error(f"Gemini prompt blocked for {image_path}: {bpe}", exc_info=True)
             # Decide how to handle: return specific message, empty string, or re-raise
             return "[Prompt Blocked by Safety Filter]"
        except Exception as e:
            self.logger.error(f"Error during Gemini OCR for {image_path}: {str(e)}", exc_info=True)
            # Re-raise a more generic exception to the caller
            raise RuntimeError(f"Failed to process prescription image via Gemini: {e}")

    # This function is now effectively replaced by image_to_text,
    # but keep it if you want a separate layer of abstraction later.
    # For now, we'll call image_to_text directly from app2.py
    # def process_prescription(self, image_path: str) -> dict:
    #     """
    #     DEPRECATED in favor of directly calling image_to_text for raw text extraction.
    #     Processes prescription image to extract raw text.
    #     """
    #     try:
    #         text = self.image_to_text(image_path)
    #         # Return only raw text as per new requirement
    #         # The dictionary structure is no longer needed here.
    #         return {'raw_text': text} # Keep dict for compatibility? Or just return text? Let's return text.

    #     except Exception as e:
    #         self.logger.error(f"Error processing prescription: {str(e)}")
    #         raise