"""
WakeSleepSTT - lightweight wake-word controlled real-time STT module (Python + Vosk)

Public API (class WakeSleepSTT):
  - constructor(config)
  - on(event, callback)   # events: 'wake','sleep','transcript','error'
  - start()               # starts mic listening (wake-word mode)
  - stop()                # stop listening & transcription
  - close()               # cleanup

Transcripts emitted as dict: { 'text': str, 'isFinal': bool, 'timestamp': float }
"""

import json
import time
import threading
import queue
from vosk import Model, KaldiRecognizer
from .audio_capture import MicrophoneStream

class WakeSleepSTT:
    def __init__(self, config=None):
        cfg = config or {}
        self.wake_words = [w.lower() for w in cfg.get('wakeWords', ['hi'])]
        self.sleep_words = [w.lower() for w in cfg.get('sleepWords', ['bye'])]
        self.model_path = cfg.get('modelPath', './models/vosk')
        self.sample_rate = cfg.get('sampleRate', 16000)
        self.audio_source = cfg.get('audioSource', 'mic')
        self._callbacks = {'wake': [], 'sleep': [], 'transcript': [], 'error': []}
        self._model = None
        self._recognizer = None
        self._mic = None
        self._running = False
        self._awake = False
        self._audio_q = queue.Queue(maxsize=200)
        self._proc_thread = None
        self._lock = threading.Lock()

    # ---- public API ----
    def on(self, event, cb):
        if event not in self._callbacks:
            raise ValueError('Unsupported event: ' + str(event))
        self._callbacks[event].append(cb)

    def start(self):
        """
        Start microphone capture and processing thread. Module enters wake-listening mode.
        """
        with self._lock:
            if self._running:
                return
            try:
                self._model = Model(self.model_path)
            except Exception as e:
                self._emit('error', e)
                raise
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
            self._recognizer.SetWords(False)  

            self._mic = MicrophoneStream(sample_rate=self.sample_rate, blocksize=4000, audio_q=self._audio_q)
            self._mic.start()
            self._running = True
            self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._proc_thread.start()

    def stop(self):
        """
        Stop mic and processing. Module will stop completely.
        """
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

    def close(self):
        self.stop()
        self._emit('closed', None)

    def _emit(self, event, payload):
        for cb in list(self._callbacks.get(event, [])):
            try:
                cb(payload)
            except Exception as e:
                try:
                    for ec in self._callbacks.get('error', []):
                        ec(e)
                except Exception:
                    pass

    def _processing_loop(self):
        """
        Consume raw PCM chunks from audio queue, feed recognizer, and handle wake/sleep logic.
        We run a single Vosk recognizer: it produces partial results (partialResult) and final results (result).
        We interpret partial + final texts for keywords.
        """
        session_active = False
        last_wake_time = 0
        debounce_secs = 0.6

        recognizer = KaldiRecognizer(self._model, self.sample_rate)
        recognizer.SetWords(False)

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                continue

            try:
                accepted = recognizer.AcceptWaveform(chunk)
            except Exception as e:
                self._emit('error', e)
                continue

            if accepted:
                try:
                    res = json.loads(recognizer.Result())
                except Exception as e:
                    self._emit('error', e)
                    continue
                text = (res.get('text') or '').strip()
                if text:
                    low = text.lower()
                    if not session_active:
                        if any(w in low for w in self.wake_words) and (time.time() - last_wake_time) > debounce_secs:
                            last_wake_time = time.time()
                            session_active = True
                            self._awake = True
                            self._emit('wake', {'word': text, 'timestamp': time.time()})
                    else:
                        self._emit('transcript', {'text': text, 'isFinal': True, 'timestamp': time.time()})
                        if any(s in low for s in self.sleep_words):
                            session_active = False
                            self._awake = False
                            self._emit('sleep', {'word': text, 'timestamp': time.time()})
            else:
                try:
                    pres = json.loads(recognizer.PartialResult())
                except Exception as e:
                    self._emit('error', e)
                    continue
                partial = pres.get('partial', '').strip()
                if partial:
                    low = partial.lower()
                    if not session_active:
                        if any(w in low for w in self.wake_words) and (time.time() - last_wake_time) > debounce_secs:
                            last_wake_time = time.time()
                            session_active = True
                            self._awake = True
                            self._emit('wake', {'word': partial, 'timestamp': time.time()})
                    else:
                        self._emit('transcript', {'text': partial, 'isFinal': False, 'timestamp': time.time()})
                        if any(s in low for s in self.sleep_words):
                            session_active = False
                            self._awake = False
                            self._emit('sleep', {'word': partial, 'timestamp': time.time()})
            time.sleep(0.001)

        try:
            self._audio_q.queue.clear()
        except Exception:
            pass
