# ---------- Imports ----------
import os
import tempfile
import time
import queue
import wave
import pyaudio
import numpy as np
import whisper
import torch
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from gtts import gTTS
import pyttsx3
from flask import Flask, request, jsonify, send_from_directory
import subprocess
import shutil

# ---------- Logging Setup ----------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Load Gemini API Key from .env file ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

# ---------- Configure Gemini Model ----------
genai.configure(api_key=GEMINI_API_KEY)
try:
    gemini_ai_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    print("Gemini model initialized successfully!")
except Exception as error:
    print(f"Error initializing Gemini model: {error}")
    exit(1)

# ---------- Validate FFmpeg Path ----------
FFMPEG_DIRECTORY = r"C:\\ffmpeg\\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_DIRECTORY

def validate_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg is properly configured!")
        return True
    except Exception:
        print("FFmpeg validation failed.")
        return False

if not validate_ffmpeg():
    exit(1)

# ---------- Load Whisper Model ----------
print("Loading Whisper model...")
device_type = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device_type)
print("Whisper model loaded successfully!")

# ---------- Audio Configuration ----------
AUDIO_FORMAT = pyaudio.paInt16      # 16-bit audio
AUDIO_CHANNELS = 1                  # Mono channel
SAMPLE_RATE = 16000                 # 16kHz sample rate
CHUNK_SIZE = 1024                   # Frames per buffer
SILENCE_THRESHOLD = 500             # Silence threshold for stopping
SILENCE_SECONDS = 2                 # Max silence before stopping

# ---------- Text-to-Speech Engine ----------
tts_fallback_engine = pyttsx3.init()

# ---------- Main Processing Class ----------
class SpeechToSpeechSystem:
    def __init__(self):
        self.is_recording = False
        self.raw_audio_data = []
        self.audio_interface = pyaudio.PyAudio()
        self.response_memory = {}
        self.MAX_MEMORY_SIZE = 1000

    # Remove old entries if cache is full
    def _purge_old_responses(self):
        if len(self.response_memory) > self.MAX_MEMORY_SIZE:
            old_keys = list(self.response_memory.keys())[:self.MAX_MEMORY_SIZE//5]
            for key in old_keys:
                del self.response_memory[key]

    # Automatically record until silence is detected
    def record_until_silence(self):
        self.is_recording = True
        self.raw_audio_data = []
        audio_stream = self.audio_interface.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

        silence_chunk_counter = 0
        silence_limit_chunks = int(SILENCE_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

        while self.is_recording:
            data = audio_stream.read(CHUNK_SIZE)
            self.raw_audio_data.append(data)
            current_chunk = np.frombuffer(data, dtype=np.int16)
            if np.abs(current_chunk).mean() < SILENCE_THRESHOLD:
                silence_chunk_counter += 1
                if silence_chunk_counter > silence_limit_chunks:
                    break
            else:
                silence_chunk_counter = 0

        audio_stream.stop_stream()
        audio_stream.close()
        self.is_recording = False

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            filename = temp_audio_file.name

        with wave.open(filename, 'wb') as wave_file:
            wave_file.setnchannels(AUDIO_CHANNELS)
            wave_file.setsampwidth(self.audio_interface.get_sample_size(AUDIO_FORMAT))
            wave_file.setframerate(SAMPLE_RATE)
            wave_file.writeframes(b''.join(self.raw_audio_data))

        return filename

    # Transcribe audio to text using Whisper
    def transcribe_audio(self, audio_file_path):
        with torch.no_grad():
            result = whisper_model.transcribe(audio_file_path, language="en", fp16=(device_type == "cuda"))
        return result["text"].strip()

    # Generate AI response using Gemini
    def generate_response_with_gemini(self, user_text, system_prompt=None, instruction_prompt=None):
        cache_id = f"{user_text}_{system_prompt}_{instruction_prompt}"
        if cache_id in self.response_memory:
            return self.response_memory[cache_id]

        try:
            full_prompt = f"{system_prompt}\n\n{user_text}\n\n{instruction_prompt}" if system_prompt and instruction_prompt else user_text
            response = gemini_ai_model.generate_content(
                full_prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            reply_text = response.text
            self.response_memory[cache_id] = reply_text
            self._purge_old_responses()
            return reply_text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "I encountered an error processing your request."

    # Convert text to speech using gTTS and fallback to pyttsx3
    def text_to_speech(self, message):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as speech_file:
                speech_filename = speech_file.name
            gTTS(text=message, lang='en', slow=False).save(speech_filename)
            return speech_filename
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fallback_file:
                speech_filename = fallback_file.name
            tts_fallback_engine.save_to_file(message, speech_filename)
            tts_fallback_engine.runAndWait()
            return speech_filename

    # Complete pipeline from audio to AI speech
    def full_process_pipeline(self, audio_input_path, system_prompt, user_instruction):
        try:
            user_text = self.transcribe_audio(audio_input_path)
            ai_reply = self.generate_response_with_gemini(user_text, system_prompt, user_instruction)
            audio_response_path = self.text_to_speech(ai_reply)
            return user_text, ai_reply, audio_response_path
        except Exception as err:
            print(f"Processing Error: {err}")
            return "", "An error occurred during processing", None

    def cleanup(self):
        self.audio_interface.terminate()

# ---------- Flask App Setup ----------
app = Flask(__name__)

# Serve the HTML interface
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Handle audio upload and process it
@app.route('/process-audio', methods=['POST'])
def process_audio_endpoint():
    try:
        os.makedirs('temp_audio', exist_ok=True)
        uploaded_audio = request.files.get('audio')

        if not uploaded_audio:
            return jsonify({'error': 'No audio file provided'}), 400

        input_filename = f"input_{int(time.time())}.wav"
        saved_audio_path = os.path.join('temp_audio', input_filename)
        uploaded_audio.save(saved_audio_path)

        system = SpeechToSpeechSystem()
        transcription, ai_message, audio_file = system.full_process_pipeline(
            saved_audio_path,
            "You are a helpful voice assistant. Respond concisely and friendly.",
            "Respond to this message"
        )

        if audio_file:
            output_filename = f"output_{int(time.time())}.wav"
            output_path = os.path.join('temp_audio', output_filename)
            shutil.copy2(audio_file, output_path)
            system.cleanup()
            return jsonify({
                'transcription': transcription,
                'aiResponse': ai_message,
                'audioOutput': f'/audio/{output_filename}'
            })
        else:
            return jsonify({'error': 'Audio generation failed'}), 500

    except Exception as e:
        logging.error(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

# Text-to-speech endpoint
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        system = SpeechToSpeechSystem()
        audio_output = system.text_to_speech(data['text'])

        if audio_output:
            output_filename = f"output_{int(time.time())}.wav"
            output_path = os.path.join('temp_audio', output_filename)
            shutil.copy2(audio_output, output_path)
            system.cleanup()
            return jsonify({
                'success': True,
                'audioOutput': f'/audio/{output_filename}'
            })
        return jsonify({'error': 'Audio generation failed'}), 500

    except Exception as e:
        logging.error(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500

# Serve audio files
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('temp_audio', filename)

# ---------- Run Flask Server ----------
if __name__ == "__main__":
    app.run(debug=True)
