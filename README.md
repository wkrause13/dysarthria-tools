# Whisper.cpp Local POC

Mac-local proof of concept for dysarthric speech transcription with `whisper.cpp`.

## What It Does

- captures microphone audio at `16 kHz` mono
- uses `webrtcvad` to detect end-of-utterance
- writes each utterance to a temp WAV file
- invokes a local `whisper.cpp` binary against a supplied model
- prints transcript and latency
- optionally speaks the text with macOS `say`

## Install

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
  --metrics-csv /Users/williamkrause/Documents/whisper-small-dysarthria-local/live-metrics.csv
```

Each completed utterance appends a CSV row with:
- speech duration
- endpointing delay
- ASR runtime
- TTS runtime
- total post-speech delay
- transcript text

## Notes

- The model path must point to a `whisper.cpp` model artifact, not the Hugging Face `model.safetensors` file.
- The current loop is utterance-based endpointing, not token streaming.
- For latency measurements, run first with `--no-tts`.
- On this Mac, Apple Command Line Tools is missing the default libc++ header symlink. The documented `CPLUS_INCLUDE_PATH=.../MacOSX.sdk/usr/include/c++/v1` workaround is required when rebuilding `whisper.cpp`.
