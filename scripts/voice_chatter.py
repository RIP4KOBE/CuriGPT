import os
import sounddevice as sd
import requests
import json
from pydub import AudioSegment
from pydub.playback import play
import wave
import base64

# Load environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to record audio from the microphone
def record_audio(duration=5, filename="output.wav", rate=44100):
    print("Recording... speak now:")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=2, dtype='int16')
    sd.wait()
    print("Recording stopped.")
    with wave.open(filename, 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.tobytes())

# Function to transcribe audio using OpenAI's Whisper
def transcribe_audio(filename="output.wav"):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "multipart/form-data"
    }
    files = {
        'file': open(filename, 'rb'),
        'model': (None, 'whisper-1'),
        'response_format': (None, 'text')
    }
    response = requests.post(url, headers=headers, files=files)
    return json.loads(response.text)

# Function to convert text to speech
def text_to_speech(text, model="tts-1", voice="echo"):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3"
    })
    response = requests.post(url, headers=headers, data=data)
    audio_content = response.content

    # Save as MP3 and convert to playable format
    with open("output.mp3", "wb") as f:
        f.write(audio_content)
    sound = AudioSegment.from_mp3("output.mp3")
    return sound

# Main interaction loop
def main():
    while True:
        record_audio()
        result = transcribe_audio()
        if 'text' in result:
            text = result['text']
            print(f"You said: {text}")
            # Simulate a reply (replace with actual AI model interaction)
            response_text = "This is a response from your AI."
            response_audio = text_to_speech(response_text)
            play(response_audio)
            print(f"Assistant said: {response_text}")
        else:
            print("Transcription failed or was unclear.")

        if input("Press Enter to continue or type 'exit' to quit: ").lower() == 'exit':
            break

if __name__ == "__main__":
    main()