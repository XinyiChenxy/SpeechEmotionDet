import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
CODES = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
LABEL_MAP = {code: {'id': i, 'emotion': LABELS[i]} for i, code in enumerate(CODES)}


def project_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    with project_path(path).open() as f:
        cfg = yaml.safe_load(f)
    if cfg['data']['sample_rate'] != 16000:
        raise ValueError('The speech encoder requires 16000 Hz input.')
    if cfg['data']['max_seconds'] is not None and int(cfg['data']['max_seconds'] * 16000) < 400:
        raise ValueError('max_seconds must provide at least 400 samples.')
    if cfg['model'].get('backend') == 'emotion2vec':
        if cfg['model']['head'] not in ('linear', 'mlp'):
            raise ValueError('head must be linear or mlp')
        if cfg['data'].get('augment', False):
            raise ValueError('Cached frozen features require augment: false.')
    elif cfg['model']['version'] not in ('Vesper-4', 'Vesper-12'):
        raise ValueError('Supported official versions: Vesper-4 / Vesper-12.')
    if cfg['train']['batch_size'] < 1 or cfg['train']['epochs'] < 1:
        raise ValueError('batch_size and epochs must be positive.')
    return cfg


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(value='auto'):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Set train.device: cpu.')
    return torch.device(value)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
