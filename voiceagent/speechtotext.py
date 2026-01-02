import asyncio
import os
from dotenv import load_dotenv
import speech_recognition as sr
from openai import OpenAI
from groq import AsyncGroq
import pyaudio

load_dotenv()

# --- 1. AUDIO PLAYER SETTINGS ---
class LocalAudioPlayer:
    def __init__(self, sample_rate=24000):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True
        )

    def write(self, chunk):
        self.stream.write(chunk)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# ------------------------------------------------------------------------

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
async_client = AsyncGroq()

async def tts(speech: str):
    # SAFETY LOCK: The Orpheus model crashes if text is > 200 chars.
    # We truncate it to 200 characters to prevent the crash.
    safe_speech = speech[:200] 
    
    try:
        async with async_client.audio.speech.with_streaming_response.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=safe_speech,
            response_format="wav"
        ) as response:
            
            player = LocalAudioPlayer(sample_rate=24000)
            async for chunk in response.iter_bytes():
                if chunk:
                    player.write(chunk)
            player.close()
            
    except Exception as e:
        print(f"Audio Error: {e}")

def main():
    r = sr.Recognizer()

    # FIX: Use a stable model system prompt
    SYSTEM_PROMPT = "You are a helpful voice assistant. You MUST keep your answers extremely short (under 2 sentences)."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1.5
        
        print("Voice Agent Ready. Speak now!")

        while True:
            print("\nListening...")
            try:
                audio = r.listen(source)
                print("Processing...")
                
                stt = r.recognize_google(audio)
                print("You said:", stt)

                messages.append({"role": "user", "content": stt})

                # FIX: Changed model to a STABLE version
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", 
                    messages=messages
                )
                
                ai_text = response.choices[0].message.content
                print("AI response:", ai_text)
                
                messages.append({"role": "assistant", "content": ai_text})
                
                # Speak Response
                asyncio.run(tts(speech=ai_text))

            except sr.UnknownValueError:
                print("...")
            except sr.RequestError:
                print("Connection error.")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()