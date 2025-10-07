"""
demo/py_demo.py

Deterministic wake/sleep demo:
 - Hotword recognizer (grammar ["hello","goodbye"]) runs while idle.
 - Start full recognizer ONLY on FINAL hotword result "hello".
 - Start full recognizer from the NEXT audio chunk (avoids retro transcripts).
 - Stop on FINAL "goodbye" from hotword OR final/partial "goodbye" from full recognizer (defensive).
 - Saves transcript only for audio spoken between hello -> goodbye.
"""
import argparse
import sounddevice as sd
import queue
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import os
from datetime import datetime
import time

q = queue.Queue()

def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)

def callback(indata, frames, time_info, status):
    if status:
        pass
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata
    q.put(mono.copy())

def save_transcript(transcript_text, folder="transcripts"):
    text = (transcript_text or "").strip()
    if not text:
        return
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"transcript_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[💾] Transcript saved to: {filename}")

def token_exact_in(text, tokens):
    """Return True if any whitespace token in text exactly equals one of tokens."""
    if not text:
        return False
    parts = text.lower().split()
    return any(p in tokens for p in parts)

def create_hot_recognizer(model, samplerate):
    grammar = json.dumps(["hello", "goodbye"])
    try:
        return KaldiRecognizer(model, samplerate, grammar)
    except Exception:
        return KaldiRecognizer(model, samplerate)

def main(model_path):
    print("Starting demo. Say 'Hello' to wake and 'Goodbye' to sleep. Press Ctrl+C to exit.")
    model = Model(model_path)

    samplerate = 16000

    
    hot_recognizer = create_hot_recognizer(model, samplerate)
    full_recognizer = None
    active = False          
    transcript_buffer = []
    debounce_secs = 0.8
    last_wake_ts = 0.0
    start_full_on_next_chunk = False

    blocksize = 4000  

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback, blocksize=blocksize):
        try:
            while True:
                data = q.get()
                if data is None:
                    continue

                int16 = float32_to_int16(data)
                chunk = int16.tobytes()

                if start_full_on_next_chunk:
                    start_full_on_next_chunk = False
                    try:
                        full_recognizer = KaldiRecognizer(model, samplerate)
                        full_recognizer.SetWords(False)
                        active = True
                        transcript_buffer = []
                        try:
                            hot_recognizer = None
                        except Exception:
                            hot_recognizer = None
                        print("[WAKE] Full recognizer started (from this chunk onward)")
                    except Exception as e:
                        print("[ERROR] failed to start full recognizer:", e)
                        full_recognizer = None
                        active = False
                if hot_recognizer is not None:
                    try:
                        hot_ok = hot_recognizer.AcceptWaveform(chunk)
                    except Exception as e:
                        print("[ERROR] hot recognizer AcceptWaveform:", e)
                        hot_ok = False

                    if hot_ok:
                        try:
                            res = json.loads(hot_recognizer.Result())
                        except Exception:
                            res = {}
                        text = (res.get("text") or "").strip().lower()
                        if text:
                            print("[HOT][FINAL]", text)
                            if not active and token_exact_in(text, {"hello"}) and (time.time() - last_wake_ts) > debounce_secs:
                                last_wake_ts = time.time()
                                start_full_on_next_chunk = True
                                print("[HOT] final 'hello' detected -> scheduling full recognizer start on next chunk")
                      
                                continue
                            if not active and token_exact_in(text, {"goodbye"}):
                                print("[HOT] final 'goodbye' detected while idle -> ignored")
                    else:
                        try:
                            pres = json.loads(hot_recognizer.PartialResult())
                        except Exception:
                            pres = {}
                        partial = (pres.get("partial") or "").strip().lower()
                        if partial:
                            if token_exact_in(partial, {"goodbye"}) and active:
                                print("[HOT][PARTIAL] goodbye (partial) -> stopping (defensive)")
                                if full_recognizer:
                                    try:
                                        final_raw = full_recognizer.Result()
                                        final_obj = json.loads(final_raw)
                                        final_text = (final_obj.get("text") or "").strip()
                                        if final_text:
                                            transcript_buffer.append(final_text)
                                    except Exception:
                                        pass
                                if transcript_buffer:
                                    save_transcript(" ".join(transcript_buffer))
                                transcript_buffer = []
                                active = False
                                full_recognizer = None
                                hot_recognizer = create_hot_recognizer(model, samplerate)
                                continue

                if active and full_recognizer is not None:
                    try:
                        ok = full_recognizer.AcceptWaveform(chunk)
                    except Exception as e:
                        print("[ERROR] full recognizer AcceptWaveform:", e)
                        ok = False

                    if ok:
                        try:
                            fres = json.loads(full_recognizer.Result())
                        except Exception:
                            fres = {}
                        ftext = (fres.get("text") or "").strip()
                        if ftext:
                            print("[TRANSCRIPT][FINAL]", ftext)
                            transcript_buffer.append(ftext)
                            if token_exact_in(ftext, {"goodbye"}):
                                print("[SLEEP] Detected 'goodbye' inside full final -> stopping")
                                if transcript_buffer:
                                    save_transcript(" ".join(transcript_buffer))
                                transcript_buffer = []
                                active = False
                                full_recognizer = None
                                hot_recognizer = create_hot_recognizer(model, samplerate)
                                continue
                    else:
                        try:
                            fpres = json.loads(full_recognizer.PartialResult())
                        except Exception:
                            fpres = {}
                        fpartial = (fpres.get("partial") or "").strip()
                        if fpartial:
                            print("[TRANSCRIPT][PART]", fpartial)
                            if token_exact_in(fpartial, {"goodbye"}):
                                print("[SLEEP] Detected 'goodbye' in full partial -> stopping (defensive)")
                                try:
                                    final_raw = full_recognizer.Result()
                                    final_obj = json.loads(final_raw)
                                    final_text = (final_obj.get("text") or "").strip()
                                    if final_text:
                                        transcript_buffer.append(final_text)
                                except Exception:
                                    pass
                                if transcript_buffer:
                                    save_transcript(" ".join(transcript_buffer))
                                transcript_buffer = []
                                active = False
                                full_recognizer = None
                                hot_recognizer = create_hot_recognizer(model, samplerate)
                                continue

                if not active and hot_recognizer is None:
                    hot_recognizer = create_hot_recognizer(model, samplerate)

        except KeyboardInterrupt:
            print("\nStopping... Goodbye 👋")
            if active and transcript_buffer:
                save_transcript(" ".join(transcript_buffer))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
