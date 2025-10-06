import argparse
import sounddevice as sd
import queue
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import os
from datetime import datetime

q = queue.Queue()

def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata
    q.put(mono.copy())

def save_transcript(transcript_text):
    """Save transcript text to a timestamped file in /transcripts"""
    folder = "transcripts"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"transcript_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(transcript_text.strip() + "\n")
    print(f"[💾] Transcript saved to: {filename}")

def main(model_path):
    print("Starting demo. Say 'Hi' to wake and 'Bye' to sleep. Press Ctrl+C to exit.")
    model = Model(model_path)
    samplerate = 44100
    recognizer = KaldiRecognizer(model, samplerate)

    active = False
    transcript_buffer = []

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback):
        try:
            while True:
                data = q.get()
                int16 = float32_to_int16(data)
                if recognizer.AcceptWaveform(int16.tobytes()):
                    res = json.loads(recognizer.Result())
                    text = res.get("text", "").lower().strip()
                    if text:
                        print("[TEXT]", text)

                        if "hi" in text and not active:
                            print("[WAKE] Detected wake word: hi")
                            active = True
                            transcript_buffer = []  
                            continue

                        if active:
                            transcript_buffer.append(text)

                        if "bye" in text and active:
                            print("[SLEEP] Detected sleep word: bye")
                            active = False
                            if transcript_buffer:
                                save_transcript(" ".join(transcript_buffer))
                            transcript_buffer = []

        except KeyboardInterrupt:
            print("\nStopping... Goodbye 👋")
            if active and transcript_buffer:
                save_transcript(" ".join(transcript_buffer))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
