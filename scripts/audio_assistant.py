import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import os
from pynput import keyboard
from pydub import AudioSegment
from pydub.playback import play
import openai
from openai import OpenAI

class AudioAssistant:
    def __init__(self, openai_api_key, base_url, user_input_filename, curigpt_output_filename):
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.user_input_filename = user_input_filename
        self.curigpt_output_filename = curigpt_output_filename

    def record_audio(self, sample_rate=44100, duration=10):
        print("Press 'Enter' to start recording...")
        audio_frames = np.zeros((int(sample_rate * duration), 1), dtype='float32')
        is_recording = False

        def on_press(key):
            nonlocal is_recording, audio_frames
            if key == keyboard.Key.enter and not is_recording:
                is_recording = True
                print("Recording... Release 'Enter' to stop.")
                # Start non-blocking recording
                sd.rec(samplerate=sample_rate, channels=1, dtype='float32', out=audio_frames, blocking=True)
                # audio_frames = sd.rec(duration * sample_rate, samplerate=sample_rate, channels=1, dtype='float32', blocking=False, device_index=device_index)
            else:
                print("Recording is already in progress...")

        # Define the callback function for key release
        def on_release(key):
            nonlocal is_recording
            if key == keyboard.Key.enter and is_recording:
                # Stop recording
                sd.stop()
                is_recording = False
                print("Recording stopped.")
                # write(user_input_filename, sample_rate, audio_frames)
                sf.write(self.user_input_filename, audio_frames, sample_rate)

                print("Audio saved as output.wav")
                return False  # Return False to stop the listener

        # Set up the listener for keyboard events
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def transcribe_audio(self):
        '''
        Transcribe the audio file using OpenAI's Whisper model
        '''
        audio_file = open(self.user_input_filename, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        print("Transcription:", transcription.text)
        return transcription.text

    def generate_response(self, text):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        print(response)
        return response.choices[0].message.content.strip()


    def text_to_speech(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )

        response.stream_to_file(self.curigpt_output_filename)
        sound = AudioSegment.from_mp3(self.curigpt_output_filename)
        play(sound)


if __name__ == '__main__':
    # Load your OpenAI API key from an environment variable
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    api_key = "sk-59XTKMjGzgbgSJjpC9D770A52eBd4d68902223561eE3F242"
    base_url = "https://www.jcapikey.com/v1"
    user_input_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/user_input.wav'
    curigpt_output_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/curigpt_output.mp3'

    assistant = AudioAssistant(api_key, base_url, user_input_filename, curigpt_output_filename)
    # assistant.record_audio()
    transcription = assistant.transcribe_audio()
    response_text = assistant.generate_response(transcription)
    assistant.text_to_speech(response_text)

