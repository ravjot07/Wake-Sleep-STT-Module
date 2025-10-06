
#  Wake-Sleep STT Module (Python)

A **reusable, wake-word controlled real-time Speech-to-Text (STT) module** built in Python using **Vosk** and **sounddevice**.  
It continuously listens for a **wake word** (e.g., “Hi”), starts live transcription when triggered, and stops when a **sleep word** (e.g., “Bye”) is spoken.

This module is designed for integration into mobile or desktop apps — e.g., React Native via a bridge — but can also run standalone on any laptop/PC.

---

##  Features

**Wake / Sleep Behavior**
- Always-on mic listening for the wake word.
- Starts real-time transcription when the wake word is detected.
- Stops transcription when the sleep word is detected.
- Supports repeated wake/sleep cycles.

 **Real-Time STT**
- Uses [Vosk](https://alphacephei.com/vosk/) for local offline speech recognition.
- Provides **incremental (partial)** and **final** transcripts via events.

**Modular API**
- App-friendly class interface (`WakeSleepSTT`) that can emit:
  - `wake`
  - `sleep`
  - `transcript`
  - `error`

 **Offline and Privacy-Friendly**
- No internet required after downloading the model.
- Works entirely on device.

---

---

##  Prerequisites

Before installation, ensure you have:
- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** and **venv**
- A **working microphone**
- Internet connection (for initial model & dependency download)

---

## Setup Instructions

###  Clone the repository
```bash
git clone https://github.com/ravjot07/wake-sleep-stt-module.git
cd wake-sleep-stt-module
```

Create and activate a virtual environment
Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell)

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Install system dependencies
(These provide audio drivers and libraries for mic capture)

```
sudo apt update
sudo apt install -y build-essential libsndfile1 libportaudio2 ffmpeg
```
 Install Python dependencies
Inside the activated virtual environment:
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
requirements.txt

sounddevice>=0.4.6
vosk>=0.3.50
numpy>=1.24
```
Download a Vosk model
Visit the Vosk Models Page
and download a small English model (recommended for demo):

Example: vosk-model-small-en-us-0.15.zip

Then unzip it inside the models directory:

```
mkdir -p models
unzip vosk-model-small-en-us-0.15.zip -d models/
mv models/vosk-model-small-en-us-0.15 models/vosk
```
Running the Demo
Option 1 — Using the provided shell script
```
chmod +x run_demo.sh
./run_demo.sh
```
```
Starting demo. Say "Hi" to wake and "Bye" to sleep. Press Ctrl+C to exit.
```
Then:

 Say “Hi” → wake event triggers
 Speak → partial + final transcripts appear in real time
 Say “Bye” → sleep event triggers and transcription pauses

### How It Works
```
MicrophoneStream continuously captures PCM16 audio from your mic and pushes chunks into a queue.
WakeSleepSTT consumes the audio chunks:
Runs Vosk’s recognizer on streaming input.
Checks partial and final texts for wake/sleep keywords.
Emits events through callbacks (on('wake', cb), on('sleep', cb), etc.).
While active, it emits transcripts (isFinal: true/false) that you can display or store.
```
