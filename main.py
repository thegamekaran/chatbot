from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app import get_response
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sunny-taiyaki-54949d.netlify.app"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(chat: ChatRequest):
    response = get_response(chat.message)
    return {"response": response}

@app.post("/voice")
def voice_endpoint(chat: ChatRequest):
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"  # Default to Rachel

    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        return {"error": "Missing ElevenLabs credentials"}

    # Step 1: Get AI response using Together API
    ai_response = get_response(chat.message)

    # Step 2: Prepare ElevenLabs API call
    eleven_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": ai_response,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.9,
            "use_speaker_boost": True
        }
    }

    # Step 3: Stream the audio properly
    response = requests.post(eleven_url, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        return {"error": f"Voice generation failed: {response.text}"}

    # Step 4: Stream audio back to frontend
    def generate_audio_stream(resp):
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                yield chunk

    return StreamingResponse(generate_audio_stream(response), media_type="audio/mpeg")
