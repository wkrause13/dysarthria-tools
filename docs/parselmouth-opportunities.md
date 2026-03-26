# Parselmouth Opportunities

Potential applications of praat-parselmouth beyond the current prosody extraction + Azure TTS integration.

## Current Project Improvements

### Acoustic Hallucination Filtering

`is_hallucination()` in `asr_whispercpp.py` is text-based. Parselmouth could add an acoustic pre-check on the utterance WAV before invoking ASR: if mean HNR is very low and no voiced frames are detected, the segment is likely noise — skip the whisper.cpp subprocess entirely. Catches hallucinations the phrase list misses and saves ASR cost.

### Dysarthria Metrics Logging

The metrics CSV already tracks timing. Add per-utterance speech quality measures:

- **Jitter** (pitch instability) — elevated in dysarthria
- **Shimmer** (amplitude instability) — elevated in dysarthria
- **HNR** (harmonics-to-noise ratio) — reduced in dysarthria
- **Mean pitch + pitch range** — already extracted in `prosody.py`

Turns the CSV into a longitudinal speech quality record for tracking progression or therapy response.

### Speaker Gate Improvement

`speaker_gate.py` uses a hand-rolled spectral embedding (hamming-windowed FFT log mean/std). Parselmouth provides classic speaker-discriminative features: formant frequencies (F1-F4), jitter/shimmer, HNR. These could replace or augment `_speaker_embedding()` for stronger verification without a heavy model dependency.

### Adaptive Endpointing

Dysarthric speakers often have longer inter-word pauses than the fixed `--silence-ms` threshold handles. Pitch contour before silence could distinguish end-of-utterance from mid-utterance pauses (falling contour = sentence boundary, level/rising = speaker still mid-thought). Hardest to tune well.

## Whisper Fine-Tuning Applications

### Data Augmentation

Dysarthric speech corpora are small. Parselmouth exposes Praat's signal manipulation to multiply training samples:

- Vary pitch contour, speaking rate, and formant structure on existing recordings
- Degrade clear speech toward dysarthric characteristics (increase jitter/shimmer, reduce HNR, compress pitch range) to generate semi-realistic training pairs
- More acoustically principled than generic augmentation (noise, time-stretch)

### Severity-Stratified Training

Parselmouth features (jitter, shimmer, HNR, pitch variability) can bin recordings by severity automatically. Enables curriculum learning — fine-tune on mild cases first where phonetic content is more recoverable, then introduce severe cases. Also ensures training set isn't skewed toward one severity level.

Depends on access to a larger, severity-diverse sample set.

### Fine-Tuning Error Analysis

After a fine-tuning round, correlate per-utterance WER with parselmouth features to find where the model breaks down. If WER spikes when HNR drops below a threshold or jitter exceeds a value, that identifies what to target next — more training data at that severity, or a different augmentation strategy.

### Speaker Normalization

Vocal tract length normalization via formant scaling (which Praat/parselmouth supports) could reduce speaker variability in training data, letting the model focus on dysarthric speech patterns rather than individual speaker characteristics.
