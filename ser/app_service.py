"""Reusable local-app inference using the validation-selected best checkpoint."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import torch

from .audio import preprocess
from .common import LABELS, ROOT, device_for, seed_all, sha256
from .emotion2vec import FrozenEncoder, load_head

BEST_CHECKPOINT = ROOT / 'outputs/cremad_emotion2vec_linear/best.pt'
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 30


def best_checkpoint_info(checkpoint=BEST_CHECKPOINT):
    checkpoint = Path(checkpoint)
    if checkpoint.name != 'best.pt':
        raise ValueError('The app only accepts the validation-selected best.pt checkpoint.')
    _, saved = load_head(checkpoint)
    history = checkpoint.parent / 'history.jsonl'
    if not history.is_file():
        raise FileNotFoundError('Missing history.jsonl; the best checkpoint cannot be verified.')
    records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    selected = max(records, key=lambda row: row['val_macro_f1'])
    if (saved['epoch'] + 1 != int(selected['epoch']) or
            abs(saved['best_macro_f1'] - selected['val_macro_f1']) > 1e-10):
        raise ValueError('best.pt does not match the best validation epoch in the training history.')
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'best_epoch': saved['epoch'] + 1, 'val_macro_f1': saved['best_macro_f1'],
            'encoder': saved['encoder']['repo'], 'head': saved['config']['model']['head'],
            'selection': 'validation Macro-F1', 'labels': LABELS}


class BestModelPredictor:
    def __init__(self, device='auto'):
        self.info = best_checkpoint_info()
        self.head, saved = load_head(BEST_CHECKPOINT)
        self.cfg = saved['config']
        seed_all(self.cfg['seed'], self.cfg['train']['torch_threads'])
        self.device = device_for(device)
        self.encoder = FrozenEncoder(self.cfg['model']['encoder_dir'], self.device, saved['encoder'])
        self.head.to(self.device).eval().requires_grad_(False)
        self.lock = threading.Lock()

    def predict_bytes(self, content, filename):
        suffix = Path(filename).suffix.lower()
        if suffix not in ('.wav', '.mp3'):
            raise ValueError('Please select a WAV or MP3 audio file.')
        if not content:
            raise ValueError('The audio file is empty. Please select another file.')
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError('The audio file exceeds the 20 MB limit.')
        started = time.monotonic()
        # Original filenames never become server filesystem paths. Uploads are
        # deleted after inference, including decode errors and failed requests.
        with tempfile.TemporaryDirectory(prefix='ser-upload-') as directory:
            path = Path(directory) / ('input' + suffix)
            path.write_bytes(content)
            try:
                waveform = preprocess(path, self.cfg['data'], augment=False)
            except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
                if 'too short' in str(error):
                    raise ValueError('The audio is too short. Please provide at least 25 milliseconds.') from error
                raise ValueError('The audio could not be decoded. Please upload a valid WAV or MP3 file.') from error
            duration = waveform.numel() / self.cfg['data']['sample_rate']
            if duration > MAX_SECONDS:
                raise ValueError('Please select a single audio clip of 30 seconds or less.')
            with self.lock, torch.inference_mode():
                feature = self.encoder.waveform(waveform)
                logits = self.head(feature[None, :].to(self.device))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('The model produced an invalid output. Please try another clip.')
                probabilities = logits.float().softmax(-1)[0].cpu().tolist()
        prediction = LABELS[max(range(6), key=probabilities.__getitem__)]
        return {'filename': Path(filename).name, 'prediction': prediction,
                'probabilities': dict(zip(LABELS, probabilities)),
                'duration_seconds': duration, 'inference_seconds': time.monotonic() - started,
                'best_epoch': self.info['best_epoch'], 'checkpoint_sha256': self.info['checkpoint_sha256'],
                'encoder': self.info['encoder'], 'head': self.info['head']}
