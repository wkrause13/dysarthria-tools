# Whisper.cpp Local POC

Mac-local proof of concept for dysarthric speech transcription with `whisper.cpp`.

## What It Does

- captures microphone audio at `16 kHz` mono
- uses `webrtcvad` to detect end-of-utterance
- writes each utterance to a temp WAV file
- invokes a local `whisper.cpp` binary against a supplied model
- prints transcript and latency
- optionally speaks the text with native macOS speech synthesis

## Install

```bash
python3 -m pip install -e '.[dev]'
```

Native macOS TTS now uses PyObjC and `NSSpeechSynthesizer`, so reinstall after pulling dependency changes:

```bash
python3 -m pip install -e '.[dev]'
```

## Convert The Fine-Tuned Model For `whisper.cpp`

Your merged Hugging Face checkpoint was converted locally and is now here:

```text
/Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin
```

The stable local `whisper.cpp` checkout and binary are here:

```text
/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src
/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli
```

If you need to rebuild `whisper.cpp` on this Mac, use:

```bash
cd /Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src
CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
  cmake -B build-local
CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
  cmake --build build-local -j --config Release
```

## Run

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary /Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli \
  --model-path /Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin
```

Useful flags:

```bash
PYTHONPATH=src python3 -m whisper_poc.app --help
```

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary /Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli \
  --model-path /Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin \
  --no-tts \
  --voice Samantha \
  --vad-mode 2 \
  --silence-ms 450
```

Recommended first live timing run:

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary /Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli \
  --model-path /Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin \
  --no-tts \
  --vad-mode 2 \
  --silence-ms 450 \
  --min-speech-ms 600 \
  --metrics-csv /Users/williamkrause/Documents/whisper-small-dysarthria-local/live-metrics.csv
```

Each completed utterance appends a CSV row with:
- speech duration
- endpointing delay
- ASR runtime
- TTS runtime
- total post-speech delay
- transcript text

Short segments below `--min-speech-ms` are ignored before transcription. For your initial live runs, `600` is a reasonable starting point to suppress spurious VAD triggers.

If background noise prevents the stop boundary from being detected, add a hard utterance cap:

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli" \
  --model-path "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin" \
  --device 2 \
  --vad-mode 3 \
  --silence-ms 300 \
  --min-speech-ms 300 \
  --max-utterance-ms 4000 \
  --min-rms 350 \
  --no-tts
```

`--max-utterance-ms` force-flushes the utterance even if noisy frames keep resetting the silence detector.
`--min-rms` rejects low-energy clips before Whisper sees them, which helps suppress hallucinated filler like `Thank you`.

If you want to filter whole utterances to one enrolled speaker before ASR, add enrollment audio and a similarity threshold:

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli" \
  --model-path "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin" \
  --device 2 \
  --vad-mode 3 \
  --silence-ms 300 \
  --min-speech-ms 300 \
  --max-utterance-ms 4000 \
  --min-rms 350 \
  --enrollment-dir "/Users/williamkrause/Documents/voice/enrollment" \
  --speaker-threshold 0.80 \
  --no-tts
```

Speaker-gate notes:
- `--enrollment-wav` can be repeated to supply specific enrollment clips
- `--enrollment-dir` loads every `*.wav` file in that directory
- the gate scores the completed utterance after VAD, duration, and RMS filtering, then skips Whisper if the score is below `--speaker-threshold`
- this is a lightweight heuristic for same-speaker matching, not diarization or overlap separation
- start with clean, close-mic enrollment clips from the intended speaker and tune `--speaker-threshold` upward if other voices still slip through

Deterministic replay run against a saved WAV:

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli" \
  --model-path "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin" \
  --input-wav "/Users/williamkrause/Documents/voice/audio_001.wav" \
  --no-tts \
  --vad-mode 2 \
  --silence-ms 450 \
  --min-speech-ms 600 \
  --metrics-csv /tmp/endpointed-replay-metrics.csv
```

Replay mode notes:
- `--input-wav` feeds a saved WAV through the same endpointed loop instead of the live microphone
- the WAV must already match the configured audio format: `16 kHz`, mono, `16-bit PCM`
- trailing silence is padded automatically so VAD can close the final utterance
- this is the easiest way to compare endpointed ASR/TTS changes on the exact same audio repeatedly

Recommended local clear-speech run with native macOS TTS enabled:

```bash
PYTHONPATH=src python3 -m whisper_poc.app \
  --whisper-binary "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whisper.cpp-src/build-local/bin/whisper-cli" \
  --model-path "/Users/williamkrause/Documents/whisper-small-dysarthria-local/whispercpp/ggml-model.bin" \
  --vad-mode 2 \
  --silence-ms 350 \
  --min-speech-ms 600 \
  --metrics-csv "/Users/williamkrause/Documents/whisper-small-dysarthria-local/live-metrics.csv"
```

Native TTS behavior:
- speech starts asynchronously instead of blocking the full app loop
- once a recognized utterance starts playing, playback is allowed to finish
- `tts_runtime_seconds` now measures TTS launch overhead, not full speech duration

## Notes

- The model path must point to a `whisper.cpp` model artifact, not the Hugging Face `model.safetensors` file.
- The current loop is utterance-based endpointing.
- For latency measurements, run first with `--no-tts`.
- On this Mac, Apple Command Line Tools is missing the default libc++ header symlink. The documented `CPLUS_INCLUDE_PATH=.../MacOSX.sdk/usr/include/c++/v1` workaround is required when rebuilding `whisper.cpp`.
