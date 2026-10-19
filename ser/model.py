"""Thin, explicit fixes around the unmodified official Vesper classes.

No fallback to random or WavLM initialization is exposed to the CLI.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

from .common import LABELS, ROOT, project_path, sha256

UPSTREAM = ROOT / 'third_party' / 'Vesper'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Official models/__init__.py imports a missing .utils; load its actual file directly.
if not (UPSTREAM / 'models' / 'vesper.py').is_file():
    raise ImportError('Missing third_party/Vesper; clone https://github.com/HappyColor/Vesper.git there.')
sys.path.insert(0, str(UPSTREAM))
official = load_source('_course_official_vesper', UPSTREAM / 'models' / 'vesper.py')
official_cfg = load_source('_course_official_model_config', UPSTREAM / 'configs' / 'model_config.py')


def model_args(cfg):
    args = dict(official_cfg._C.Vesper)
    args.update(encoder_layers=int(cfg['version'].split('-')[1]),
                projector_dim=cfg['projector_dim'], num_classes=6,
                output_rep=cfg['output_rep'], freeze_upstream=cfg['freeze_upstream'],
                freeze_cnn=cfg['freeze_cnn'], normalize=False,
                init_with_wavlm=False, init_with_ckpt=False)
    # normalize=False because the SAME official layer_norm runs per utterance before padding.
    return SimpleNamespace(**args)


def feature_lengths(lengths):
    # Exact official valid convolutions; do not approximate by reshaping a sample mask.
    for kernel, stride in [(10, 5)] + [(3, 2)] * 4 + [(2, 2)] * 2:
        lengths = (lengths - kernel) // stride + 1
    return lengths


class PaddingSafeVesper(official.Vesper):
    def forward_padding_mask(self, features, padding_mask):
        lengths = feature_lengths((~padding_mask).sum(-1))
        if (lengths <= 0).any():
            raise ValueError('Every utterance must have at least 400 valid audio samples.')
        self.padding_mask = torch.arange(features.shape[1], device=features.device)[None, :] >= lengths[:, None]


class EmotionModel(official.VesperFinetuneWrapper):
    def __init__(self, cfg):
        # Keep official representation weighting and classifier structure; the public
        # constructors below strictly load all encoder tensors before returning.
        nn.Module.__init__(self)
        args = model_args(cfg)
        self.vesper = PaddingSafeVesper(args)
        self.vesper.mask_emb = None  # Same as official downstream need_mask_emb=False.
        self.projector = nn.Linear(args.encoder_embed_dim, args.projector_dim)
        self.classifier = nn.Linear(args.projector_dim, 6)
        self.freeze_upstream = args.freeze_upstream
        if args.output_rep not in ('weighted_sum', 'last_layer'):
            raise ValueError(f'Unsupported representation: {args.output_rep}')
        self.weights = nn.Parameter(torch.zeros(args.encoder_layers + 1)) if args.output_rep == 'weighted_sum' else None
        for p in self.vesper.parameters():
            p.requires_grad = not args.freeze_upstream
        if args.freeze_cnn:
            for p in self.vesper.feature_extractor.parameters():
                p.requires_grad = False
        self.provenance = None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_upstream:
            self.vesper.eval()  # Deterministic frozen features; no dropout noise.
        return self

    def forward(self, waveform, padding_mask=None):
        with torch.no_grad() if self.freeze_upstream else contextlib.nullcontext():
            features, layer_results = self.vesper(waveform, padding_mask, ret_layer_results=True)
        if self.weights is not None:
            features = self._weighted_sum([result[0] for result in layer_results])
        features = self.projector(features)
        mask = self.vesper.get_padding_mask()
        if mask is None:
            pooled = features.mean(1)
        else:
            # Mask AFTER the biased projector so padded positions cannot add bias.
            pooled = features.masked_fill(mask[..., None], 0).sum(1) / (~mask).sum(1, keepdim=True)
        return self.classifier(pooled)


def require_pretrained(path):
    path = project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Genuine pretrained Vesper checkpoint missing: {path}. '
            'Official repository currently provides no verified download. '
            'Supply --checkpoint /path/to/author_Vesper-4.pt; random/WavLM fallback is disabled.')
    if path.stat().st_size < 1024:
        raise ValueError(f'Checkpoint too small (possibly LFS/HTML stub): {path}')
    return path


def load_pretrained(cfg, path=None):
    path = require_pretrained(path or cfg['checkpoint'])
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(data, dict) or 'model' not in data:
        raise ValueError('Expected official checkpoint dictionary containing model (vesper.* state_dict).')
    state = {k.removeprefix('module.'): v for k, v in data['model'].items()}
    encoder = {k[len('vesper.'):]: v for k, v in state.items() if k.startswith('vesper.')}
    if not encoder:
        raise ValueError('No vesper.* encoder tensors. A WavLM state_dict is not a Vesper checkpoint.')
    layers = {int(k.split('.')[2]) for k in encoder if k.startswith('encoder.layers.')}
    expected_layers = int(cfg['version'].split('-')[1])
    if layers != set(range(expected_layers)):
        raise ValueError(f'Checkpoint layers {sorted(layers)} do not match {cfg["version"]}.')
    # Exactly the three exclusions in official init_with_ckpt, need_mask_emb=False.
    for key in ('mask_emb', 'encoder.layer_norm.weight', 'encoder.layer_norm.bias'):
        encoder.pop(key, None)
    model = EmotionModel(cfg)
    model.vesper.load_state_dict(encoder, strict=True)  # Includes full shape/config verification.
    model.provenance = {'path': str(path), 'sha256': sha256(path), 'version': cfg['version'],
                        'encoder_keys_loaded': len(encoder), 'strict_load': True,
                        'note': 'User-supplied official-format Vesper; provenance must be checked at source.'}
    return model


def load_trained(path):
    path = project_path(path)
    data = torch.load(path, map_location='cpu', weights_only=False)
    if data.get('format') != 'vesper-cremad-course-v1' or data.get('labels') != LABELS:
        raise ValueError('Not a supported course checkpoint, or label mapping mismatch.')
    if not data.get('pretrained', {}).get('strict_load'):
        raise ValueError('Checkpoint has no record of a strict pretrained encoder load.')
    model = EmotionModel(data['config']['model'])
    model.load_state_dict(data['model'], strict=True)
    model.provenance = data['pretrained']
    return model, data
