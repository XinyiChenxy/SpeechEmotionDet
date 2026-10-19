from math import gcd
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.nn.functional as F


def load_audio(path, sample_rate=16000):
    """Decode WAV/MP3, mix channels, resample. Reject LFS stubs explicitly."""
    path = Path(path)
    with path.open('rb') as f:
        if f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Git LFS pointer, not audio: {path}; run git lfs pull.')
    try:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
    except (RuntimeError, sf.LibsndfileError):
        # ffmpeg supports MP3 even on libsndfile builds without MPEG decoding.
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path),
             '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sample_rate), '-ac', '1', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=True)
        audio = np.frombuffer(result.stdout, dtype='<f4').copy()
        rate = sample_rate
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    if rate != sample_rate:
        divisor = gcd(int(rate), sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, int(rate) // divisor).astype('float32')
    return torch.from_numpy(np.ascontiguousarray(audio)), int(rate)


def preprocess(path, cfg, augment=False):
    waveform, _ = load_audio(path, cfg['sample_rate'])
    # emotion2vec uses complete utterances by default; legacy Vesper uses 3 s.
    if cfg['max_seconds'] is not None:
        waveform = waveform[:int(cfg['max_seconds'] * cfg['sample_rate'])]
    if waveform.numel() < 400:
        raise ValueError(f'Audio too short for speech CNN (minimum 400 samples): {path}')
    if augment and cfg.get('augment', False):
        low, high = cfg.get('noise_snr_db', [20.0, 35.0])
        snr = torch.empty(()).uniform_(low, high)
        noise = torch.randn_like(waveform)
        rms = waveform.square().mean().sqrt()
        waveform = waveform + noise * rms / (10 ** (snr / 20))
    # Per-utterance official layer_norm, applied BEFORE padding. No fitted statistics.
    return F.layer_norm(waveform, (waveform.numel(),))


def collate(batch):
    waveforms, labels, rows = zip(*batch)
    lengths = torch.tensor([x.numel() for x in waveforms], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    mask = torch.arange(padded.shape[1])[None, :] >= lengths[:, None]
    return {'waveform': padded, 'padding_mask': mask,
            'label': torch.tensor(labels, dtype=torch.long), 'rows': list(rows)}
