"""
WakeSleepSTT - lightweight wake/sleep word controlled real-time STT module (Python + Vosk)

Wake word: "hello"
Sleep word: "goodbye"

Behavior:
 - Hotword recognizer (grammar-limited) runs while idle.
 - On final "hello" -> wait for next voiced chunk, then start full recognizer.
 - Full recognizer runs only while awake; stops on "goodbye".
 - Only transcribes between hello → goodbye, saving transcript on sleep.
"""

import json
import time
import threading
import queue
import os
from datetime import datetime
import numpy as np
from vosk import Model, KaldiRecognizer
from .audio_capture import MicrophoneStream

class WakeSleepSTT:
    def __init__(self, config=None):
        cfg = config or {}
        self.wake_words = [w.lower() for w in cfg.get('wakeWords', ['hello'])]
        self.sleep_words = [w.lower() for w in cfg.get('sleepWords', ['goodbye'])]
        self.model_path = cfg.get('modelPath', './models/vosk')
        self.sample_rate = cfg.get('sampleRate', 16000)
        self.audio_source = cfg.get('audioSource', 'mic')
        self.vad_threshold = cfg.get('vadThreshold', 0.01)  
        self.debounce_secs = cfg.get('debounceSecs', 0.6)

        self._callbacks = {'wake': [], 'sleep': [], 'transcript': [], 'error': []}
        self._model = None
        self._hot_recognizer = None
        self._full_recognizer = None
        self._mic = None
        self._running = False
        self._awake = False
        self._audio_q = queue.Queue(maxsize=500)
        self._proc_thread = None
        self._lock = threading.Lock()
        self._session_buffer = []
        self._waiting_for_voiced_chunk = False
        self._last_wake_time = 0.0

    # ---- Public API ----
    def on(self, event, cb):
        if event not in self._callbacks:
            raise ValueError(f'Unsupported event: {event}')
        self._callbacks[event].append(cb)

    def start(self):
        """Start microphone capture and processing thread."""
        with self._lock:
            if self._running:
                return
            try:
                self._model = Model(self.model_path)
            except Exception as e:
                self._emit('error', e)
                raise

            self._hot_recognizer = self._create_hot_recognizer(self._model, self.sample_rate)
            self._mic = MicrophoneStream(sample_rate=self.sample_rate, blocksize=4000, audio_q=self._audio_q)
            self._mic.start()

            self._running = True
            self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._proc_thread.start()

    def stop(self):
        """Stop mic and processing."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._mic:
                self._mic.stop()
                self._mic = None
            if self._proc_thread:
                self._proc_thread.join(timeout=1.0)
                self._proc_thread = None

            self._hot_recognizer = None
            self._full_recognizer = None
            self._awake = False
            self._waiting_for_voiced_chunk = False
            self._session_buffer = []

    def close(self):
        self.stop()
        self._emit('closed', None)

    def _emit(self, event, payload):
        for cb in list(self._callbacks.get(event, [])):
            try:
                cb(payload)
            except Exception as e:
                for ec in self._callbacks.get('error', []):
                    try:
                        ec(e)
                    except Exception:
                        pass

    def _create_hot_recognizer(self, model, samplerate):
        grammar = json.dumps(self.wake_words + self.sleep_words)
        try:
            return KaldiRecognizer(model, samplerate, grammar)
        except Exception:
            return KaldiRecognizer(model, samplerate)

    def _save_transcript(self, text, folder="transcripts"):
        text = (text or '').strip()
        if not text:
            return
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(folder, f'transcript_{timestamp}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text + "\n")
        print(f"[💾] Transcript saved to: {filename}")

    @staticmethod
    def _rms_from_int16_bytes(bts):
        try:
            arr = np.frombuffer(bts, dtype=np.int16).astype(np.float32)
            if arr.size == 0:
                return 0.0
            arr = arr / 32767.0
            return float(np.sqrt(np.mean(np.square(arr))))
        except Exception:
            return 0.0

    def _is_wake_in(self, text):
        return any(w in text.lower() for w in self.wake_words)

    def _is_sleep_in(self, text):
        return any(s in text.lower() for s in self.sleep_words)

    def _processing_loop(self):
        session_active = False
        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                continue

            rms = self._rms_from_int16_bytes(chunk)
            if self._waiting_for_voiced_chunk:
                if rms >= self.vad_threshold:
                    try:
                        self._full_recognizer = KaldiRecognizer(self._model, self.sample_rate)
                        self._full_recognizer.SetWords(False)
                        self._awake = True
                        session_active = True
                        self._session_buffer = []
                        self._waiting_for_voiced_chunk = False
                        self._hot_recognizer = None
                        self._emit('wake', {'word': 'hello', 'timestamp': time.time()})
                    except Exception as e:
                        self._emit('error', e)
                        self._waiting_for_voiced_chunk = False
                        continue
                else:
                    continue
            if session_active and self._full_recognizer is not None:
                try:
                    accepted = self._full_recognizer.AcceptWaveform(chunk)
                except Exception as e:
                    self._emit('error', e)
                    accepted = False

                if accepted:
                    res = json.loads(self._full_recognizer.Result())
                    text = (res.get('text') or '').strip()
                    if text:
                        self._session_buffer.append(text)
                        self._emit('transcript', {'text': text, 'isFinal': True, 'timestamp': time.time()})
                        if self._is_sleep_in(text):
                            self._awake = False
                            session_active = False
                            if self._session_buffer:
                                self._save_transcript(" ".join(self._session_buffer))
                            self._session_buffer = []
                            self._full_recognizer = None
                            self._hot_recognizer = self._create_hot_recognizer(self._model, self.sample_rate)
                            self._emit('sleep', {'word': text, 'timestamp': time.time()})
                else:
                    pres = json.loads(self._full_recognizer.PartialResult())
                    partial = (pres.get('partial') or '').strip()
                    if partial:
                        self._emit('transcript', {'text': partial, 'isFinal': False, 'timestamp': time.time()})
                        if self._is_sleep_in(partial):
                            self._awake = False
                            session_active = False
                            if self._session_buffer:
                                self._save_transcript(" ".join(self._session_buffer))
                            self._session_buffer = []
                            self._full_recognizer = None
                            self._hot_recognizer = self._create_hot_recognizer(self._model, self.sample_rate)
                            self._emit('sleep', {'word': partial, 'timestamp': time.time()})
                time.sleep(0.001)
                continue
            if self._hot_recognizer is not None:
                try:
                    hot_ok = self._hot_recognizer.AcceptWaveform(chunk)
                except Exception as e:
                    self._emit('error', e)
                    hot_ok = False

                if hot_ok:
                    res = json.loads(self._hot_recognizer.Result())
                    text = (res.get('text') or '').strip().lower()
                    if text:
                        if (not session_active) and self._is_wake_in(text) and (time.time() - self._last_wake_time) > self.debounce_secs:
                            self._last_wake_time = time.time()
                            self._waiting_for_voiced_chunk = True

            time.sleep(0.001)

        if self._awake and self._session_buffer:
            self._save_transcript(" ".join(self._session_buffer))
        try:
            self._audio_q.queue.clear()
        except Exception:
            pass
