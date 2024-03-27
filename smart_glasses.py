import pyaudio
import whisper
import torch 
import numpy as np
import time 
from mesa import Agent, Model 
from mesa.time import SimultaneousActivation


class MicrophoneAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.stream = self.model.audio_stream

    def step(self):
        frames = [self.stream.read(self.model.buffer_size) for _ in range(int(self.model.sample_rate / self.model.buffer_size * 5))]
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        self.model.put_audio_data(audio_data)

class TranscriptionAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_audio_timestamp = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = whisper.load_model("tiny.en") 

    def step(self):
        audio_data = self.model.get_audio_data()

        current_timestamp = time.time()
        if current_timestamp - self.last_audio_timestamp > 1:
            if audio_data is not None:
                audio_data_float = audio_data.astype(np.float32) / 32767.0
                audio_tensor = torch.from_numpy(audio_data_float)
                result = self.whisper_model.transcribe(audio_tensor)

                transcription = result["text"]
                if transcription:
                    self.model.put_transcription(transcription)

            self.last_audio_timestamp = current_timestamp

class DisplayAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.silence_threshold = 1000

    def step(self):
        transcription = self.model.get_transcription()
        audio_data = self.model.get_audio_data()

        if audio_data is not None and np.max(np.abs(audio_data)) > self.silence_threshold:
            if transcription:
                print("Live Display:", transcription)

class SmartGlassesModel(Model):
    def __init__(self):
        super().__init__()
        print("Booting up...")
        self.buffer_size = 2048
        self.sample_rate = 12000
        self.min_transcription_length = 20
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size
        )
        self.audio_data = None
        self.transcription = None

        self.microphone_agent = MicrophoneAgent(1, self)
        self.transcription_agent = TranscriptionAgent(2, self)
        self.display_agent = DisplayAgent(3, self)
        
        self.schedule = SimultaneousActivation(self)
        self.schedule.add(self.microphone_agent)
        self.schedule.add(self.transcription_agent)
        self.schedule.add(self.display_agent)
        print("transcribing...")

    def step(self):
        self.schedule.step()

    def get_audio_data(self):
        return self.audio_data

    def put_audio_data(self, audio_data):
        self.audio_data = audio_data

    def get_transcription(self):
        return self.transcription

    def put_transcription(self, transcription):
        self.transcription = transcription

    def cleanup(self):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio_instance.terminate()

smart_glasses_model = SmartGlassesModel()

try:
    while True:
        smart_glasses_model.step()

except KeyboardInterrupt:
    print('Smart Glasses terminated by the user.')
    smart_glasses_model.cleanup()
