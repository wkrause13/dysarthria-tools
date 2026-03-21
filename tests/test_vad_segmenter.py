from whisper_poc.vad_segmenter import VadSegmenter


def test_segmenter_emits_utterance_after_configured_silence():
    frame = b"\x01\x02" * 480
    segmenter = VadSegmenter(sample_rate=16000, frame_ms=30, silence_ms=300)

    utterance = None
    for _ in range(4):
        utterance = segmenter.push(frame, is_speech=True)
        assert utterance is None

    for _ in range(9):
        utterance = segmenter.push(frame, is_speech=False)
        assert utterance is None

    utterance = segmenter.push(frame, is_speech=False)

    assert utterance is not None
    assert utterance.frame_count == 4
    assert utterance.audio_bytes == frame * 4


def test_segmenter_reports_trailing_silence_frames_on_emit():
    frame = b"\x01\x02" * 480
    segmenter = VadSegmenter(sample_rate=16000, frame_ms=30, silence_ms=300)

    for _ in range(3):
        assert segmenter.push(frame, is_speech=True) is None

    utterance = None
    for _ in range(10):
        utterance = segmenter.push(frame, is_speech=False)
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.trailing_silence_frames == 10
