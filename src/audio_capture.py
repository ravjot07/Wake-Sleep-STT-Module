import sounddevice as sd
import threading
import numpy as np
import traceback

class MicrophoneStream:
    def __init__(self, sample_rate=16000, blocksize=4000, audio_q=None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.stream = None
        self._running = False
        self.audio_q = audio_q
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        try:
            if indata.dtype != 'int16':
                audio = (indata * 32767).astype('int16')
            else:
                audio = indata
            self.audio_q.put(audio.tobytes(), block=False)
        except Exception:
            try:
                self.audio_q.put(indata.tobytes(), block=False)
            except Exception:
                pass

    def start(self):
        with self._lock:
            if self._running:
                return
            try:
                self.stream = sd.RawInputStream(samplerate=self.sample_rate,
                                               blocksize=self.blocksize,
                                               dtype='int16',
                                               channels=1,
                                               callback=self._callback)
                self.stream.start()
                self._running = True
            except Exception as e:
                raise

    def stop(self):
        with self._lock:
            if not self._running:
                return
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
            finally:
                self._running = False
