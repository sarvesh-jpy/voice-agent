import asyncio
import os
import json
import subprocess
import requests
import speech_recognition as sr
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
from groq import AsyncGroq
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables
load_dotenv()

# --- 1. SETUP CLIENTS ---
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# --- 2. AUDIO PLAYER CLASS ---
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

# --- 3. ASYNC TTS FUNCTION ---
async def tts(speech: str):
    # Safety: Truncate to avoid model crash on long text
    safe_speech = speech[:200]
    try:
        async with async_client.audio.speech.with_streaming_response.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=safe_speech,
            response_format="wav"
        ) as response:
            player = LocalAudioPlayer(sample_rate=24000)
            # Skip WAV header (44 bytes) to prevent static "pop"
            header_skipped = False
            async for chunk in response.iter_bytes():
                if not header_skipped and len(chunk) > 44:
                    player.write(chunk[44:])
                    header_skipped = True
                else:
                    player.write(chunk)
            player.close()
    except Exception as e:
        print(f"Audio Error: {e}")

# --- 4. DATA MODEL & TOOLS ---
class StepModel(BaseModel):
    step: str = Field(..., description="The step name: 'plan', 'tool', or 'output'")
    content: Optional[str] = Field(None, description="The reasoning or final message")
    tool: Optional[str] = Field(None, description="The name of the tool to call")
    input: Optional[str] = Field(None, description="The input argument for the tool")

def run_command(cmd: str):
    try:
        print(f"Executing system command: {cmd}")
        # Run command and capture output
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return result.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output.decode('utf-8')}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, headers={"User-Agent": "curl/7.68.0"}, timeout=10)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        return "Error: Could not get weather."
    except Exception as e:
        return f"Error: {str(e)}"

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

# --- 5. MAIN LOGIC ---
def main():
    SYSTEM_PROMPT = """
    You are a Voice AI Agent capable of executing tools.
    You MUST respond with valid JSON that matches this structure:
    {
      "step": "plan" | "tool" | "output",
      "content": "string",
      "tool": "tool_name", 
      "input": "tool_input"
    }

    Available Tools:
    - get_weather(city)
    - run_command(cmd) : Execute a shell command on the user's computer.

    Rules:
    1. First, PLAN what to do.
    2. If needed, use a TOOL.
    3. Finally, provide an OUTPUT to speak to the user.
    """

    print("--- 🎙️ Voice Agent Started ---")
    
    # Initialize History
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Initialize Speech Recognizer
    r = sr.Recognizer()
    
    # Open Microphone ONCE and keep it open
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1.5
        print("🟢 Ready! Speak now...")

        while True:
            try:
                print("\n👂 Listening...")
                audio = r.listen(source)
                print("🧠 Processing Audio...")
                
                # 1. Speech to Text
                user_query = r.recognize_google(audio)
                print(f"🗣️ You said: {user_query}")
                
                # Add User Query to History
                message_history.append({"role": "user", "content": user_query})

                # 2. Agent Reasoning Loop (Repeat until 'output' step)
                while True:
                    # Call LLM
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        response_format={"type": "json_object"}, 
                        messages=message_history
                    )
                    
                    raw_json_str = response.choices[0].message.content
                    
                    # Validate JSON
                    try:
                        parsed_result = StepModel.model_validate_json(raw_json_str)
                    except Exception as e:
                        print(f"⚠️ JSON Parse Error: {e}")
                        break # Break inner loop to listen again

                    # Add Assistant's thought to history
                    message_history.append({"role": "assistant", "content": raw_json_str})
                    
                    step_type = parsed_result.step.lower()

                    # --- PLAN ---
                    if step_type == "plan":
                        print(f"📝 PLAN: {parsed_result.content}")
                        # Loop again to let AI execute the plan (it will output 'tool' next)
                        continue

                    # --- TOOL ---
                    elif step_type == "tool":
                        tool_name = parsed_result.tool
                        tool_in = parsed_result.input
                        
                        print(f"🛠️ TOOL: {tool_name}('{tool_in}')")
                        
                        if tool_name in available_tools:
                            tool_output = available_tools[tool_name](tool_in)
                        else:
                            tool_output = "Error: Tool not found"
                        
                        print(f"👀 RESULT: {tool_output}")
                        
                        # Add Observation to history so AI knows the result
                        obs_json = json.dumps({"step": "observe", "content": tool_output})
                        message_history.append({"role": "user", "content": obs_json})
                        # Loop again to let AI process the observation
                        continue

                    # --- OUTPUT (Final Answer) ---
                    elif step_type == "output":
                        final_response = parsed_result.content
                        print(f"🤖 AI: {final_response}")
                        
                        # Speak Response
                        asyncio.run(tts(speech=final_response))
                        
                        # Break inner loop to go back to listening for new user input
                        break 

            except sr.UnknownValueError:
                print("... (Silence) ...")
            except sr.RequestError:
                print("❌ Connection error with Speech Service.")
            except Exception as e:
                print(f"❌ Error: {e}")
                # Reset history on critical error to avoid loop
                message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

if __name__ == "__main__":
    main()