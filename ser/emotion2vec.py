"""Frozen official emotion2vec_base -> cached utterance embeddings -> six-class head."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .audio import preprocess
from .common import (LABELS, LABEL_MAP, atomic_save, device_for, load_config, project_path,
                     restore_rng, rng_state, save_json, seed_all, sha256)
from .data import prepare, read_manifest
from .metrics import save_evaluation, scores
from .experiment_log import ExperimentLog, export_history

REPO = 'emotion2vec/emotion2vec_base'
REVISION = '0c9a3152734f9d7a7a05b4ee6bfb3c109d288664'
WEIGHT_SHA256 = '4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd'
FORMAT = 'emotion2vec-cremad-course-v1'
RECIPE = 'mono16k; per-utterance layer_norm; official extract_features(mask=False); mean last-layer valid frames'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def encoder_info(directory):
    directory = project_path(directory)
    path = directory / 'emotion2vec_base.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; run python scripts/download_emotion2vec.py first.')
    digest = sha256(path)
    if digest != WEIGHT_SHA256:
        raise ValueError(f'emotion2vec_base weight SHA256 mismatch: {path}')
    from omegaconf import OmegaConf
    model_config = OmegaConf.to_container(OmegaConf.load(directory / 'config.yaml'), resolve=True)
    return {'repo': REPO, 'revision': REVISION, 'weight_sha256': digest,
            'model_config': model_config, 'funasr_version': importlib.metadata.version('funasr'),
            'recipe': RECIPE, 'strict_load': True}


class FrozenEncoder:
    def __init__(self, directory, device, expected=None):
        self.info = encoder_info(directory)
        if self.info['funasr_version'] != '1.2.6':
            raise ValueError('Use the pinned funasr==1.2.6 environment to reproduce feature extraction.')
        if expected is not None and self.info != expected:
            raise ValueError('Encoder weights/config/runtime differ from the trained head.')
        from funasr.models.emotion2vec.model import Emotion2vec
        self.model = Emotion2vec(**self.info['model_config'])
        state = torch.load(project_path(directory) / 'emotion2vec_base.pt', map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)
        del state
        self.model.requires_grad_(False).eval().to(device)
        self.device = device

    @torch.inference_mode()
    def waveform(self, waveform):
        # Single-utterance extraction matches the official inference routine.
        # No batching/padding enters the CNN, positional encoder, or pooling.
        result = self.model.extract_features(waveform[None, :].to(self.device), padding_mask=None, mask=False)
        frames = result['x']
        if frames.ndim != 3 or frames.shape[0] != 1 or frames.shape[-1] != 768 or frames.shape[1] < 1:
            raise ValueError(f'Unexpected official feature shape: {tuple(frames.shape)}')
        embedding = frames.float().mean(1)[0].cpu().clone()
        if not torch.isfinite(embedding).all():
            raise FloatingPointError('Non-finite emotion2vec embedding')
        return embedding

    def audio(self, path, cfg):
        return self.waveform(preprocess(project_path(path), cfg, augment=False))


def cache_identity(cfg, rows, info):
    # Protect against stale features when the manifest, audio, weights, or
    # preprocessing change. No statistics are fitted during extraction.
    source_stats = []
    for row in rows:
        stat = Path(row['path']).stat()
        source_stats.append((row['path'], stat.st_size, stat.st_mtime_ns))
    return {'manifest_sha256': sha256(project_path(cfg['data']['manifest'])),
            'source_stats_sha256': fingerprint(source_stats), 'encoder': info,
            'preprocessing': {'sample_rate': cfg['data']['sample_rate'],
                              'max_seconds': cfg['data']['max_seconds'], 'recipe': RECIPE}}


def validate_cache(cache, cfg, rows, info=None):
    info = info or cache['identity']['encoder']
    expected = cache_identity(cfg, rows, info)
    if cache.get('format') != 'emotion2vec-features-v1' or cache['identity'] != expected or cache['rows'] != rows:
        raise ValueError('Feature cache is stale. Run extract --force with the intended configuration.')
    x = cache['features']
    if tuple(x.shape) != (len(rows), cfg['model']['feature_dim']) or not torch.isfinite(x).all():
        raise ValueError('Invalid feature cache shape/values')
    expected_labels = torch.tensor([int(row['label']) for row in rows])
    if not torch.equal(cache['labels'], expected_labels):
        raise ValueError('Feature cache labels differ from the manifest.')


def extract(cfg, force=False):
    seed_all(cfg['seed'], cfg['train']['torch_threads'])
    rows = read_manifest(cfg['data']['manifest'])
    info = encoder_info(cfg['model']['encoder_dir'])
    identity = cache_identity(cfg, rows, info)
    path = project_path(cfg['features']['cache'])
    if path.exists() and not force:
        cache = torch.load(path, map_location='cpu', weights_only=False)
        validate_cache(cache, cfg, rows, info)
        print(f'Verified existing cache: {path}', flush=True)
        return cache
    encoder = FrozenEncoder(cfg['model']['encoder_dir'], device_for(cfg['train']['device']), info)
    # Resumable per-utterance cache, namespaced by the complete extraction recipe.
    shard_dir = path.parent / 'utterances' / fingerprint(identity)
    shard_dir.mkdir(parents=True, exist_ok=True)
    features = []
    started = time.monotonic()
    computed = 0
    for row in tqdm(rows, desc='Frozen emotion2vec features', mininterval=3):
        shard = shard_dir / (fingerprint(row['path']) + '.npy')
        if shard.exists():
            vector = torch.from_numpy(np.load(shard, allow_pickle=False).copy())
            if vector.shape != (768,) or not torch.isfinite(vector).all():
                raise ValueError(f'Corrupt feature shard: {shard}')
        else:
            vector = encoder.audio(row['path'], cfg['data'])
            temp = shard.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.save(f, vector.numpy(), allow_pickle=False)
            temp.replace(shard)
            computed += 1
        features.append(vector)
    cache = {'format': 'emotion2vec-features-v1', 'identity': identity, 'rows': rows,
             'features': torch.stack(features), 'labels': torch.tensor([int(r['label']) for r in rows])}
    atomic_save(path, cache)
    save_json(path.with_suffix('.json'), {'identity': identity, 'samples': len(rows), 'feature_dim': 768,
                                        'seconds': time.monotonic() - started, 'newly_computed': computed,
                                        'encoder_frozen': True, 'encoder_training': encoder.model.training})
    print(f'Saved frozen features: {path}; {time.monotonic() - started:.1f} seconds', flush=True)
    return cache


def load_cache(cfg):
    path = project_path(cfg['features']['cache'])
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run python -m ser extract first.')
    cache = torch.load(path, map_location='cpu', weights_only=False)
    rows = read_manifest(cfg['data']['manifest'])
    validate_cache(cache, cfg, rows)
    return cache


def indices_for(rows, split):
    return torch.tensor([i for i, row in enumerate(rows) if row['split'] == split], dtype=torch.long)


def fit_standardizer(features, train_indices, enabled=True):
    # This function is intentionally given ONLY training indices.
    if enabled:
        train = features[train_indices].float()
        return train.mean(0), train.std(0, unbiased=False).clamp_min(1e-6)
    return torch.zeros(features.shape[1]), torch.ones(features.shape[1])


class Head(nn.Module):
    def __init__(self, cfg, mean, scale):
        super().__init__()
        self.register_buffer('feature_mean', mean.clone())
        self.register_buffer('feature_scale', scale.clone())
        dim = cfg['feature_dim']
        if cfg['head'] == 'linear':
            self.network = nn.Linear(dim, 6)
        elif cfg['head'] == 'mlp':
            self.network = nn.Sequential(nn.Linear(dim, cfg['hidden_dim']), nn.ReLU(),
                                         nn.Dropout(cfg['dropout']), nn.Linear(cfg['hidden_dim'], 6))
        else:
            raise ValueError('head must be linear or mlp')

    def forward(self, features):
        return self.network((features - self.feature_mean) / self.feature_scale)


def pack(head, cfg, cache, **extra):
    return {'format': FORMAT, 'config': copy.deepcopy(cfg), 'labels': LABELS, 'label_mapping': LABEL_MAP,
            'head': head.state_dict(), 'encoder': cache['identity']['encoder'],
            'feature_identity': cache['identity'], 'preprocessing': cache['identity']['preprocessing'], **extra}


def load_head(path):
    data = torch.load(project_path(path), map_location='cpu', weights_only=False)
    if data.get('format') != FORMAT or data.get('labels') != LABELS or data.get('label_mapping') != LABEL_MAP:
        raise ValueError('Invalid emotion2vec course head or label mapping')
    state = data['head']
    head = Head(data['config']['model'], state['feature_mean'], state['feature_scale'])
    head.load_state_dict(state, strict=True)
    return head, data


@torch.no_grad()
def score_head(head, features, targets):
    head.eval()
    logits = head(features)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite head predictions')
    probabilities = logits.softmax(-1).cpu().numpy()
    result = scores(targets.cpu().tolist(), probabilities.argmax(-1).tolist())
    result['loss'] = float(nn.functional.cross_entropy(logits, targets))
    result['samples'] = len(targets)
    return result, probabilities


def evaluate(checkpoint, split='test', output=None, device='auto', cache_override=None):
    head, saved = load_head(checkpoint)
    cfg = copy.deepcopy(saved['config'])
    seed_all(cfg['seed'], cfg['train']['torch_threads'])
    if cache_override:
        cfg['features']['cache'] = cache_override
    cache = load_cache(cfg)
    if cache['identity'] != saved['feature_identity']:
        raise ValueError('Feature cache does not match training checkpoint')
    indices = indices_for(cache['rows'], split)
    device = device_for(device)
    head.to(device)
    metrics, probabilities = score_head(head, cache['features'][indices].to(device), cache['labels'][indices].to(device))
    metrics.update(split=split, checkpoint=str(project_path(checkpoint)), checkpoint_sha256=sha256(project_path(checkpoint)),
                   manifest_sha256=cache['identity']['manifest_sha256'],
                   encoder=REPO, head=cfg['model']['head'])
    records = [cache['rows'][i] for i in indices.tolist()]
    destination = project_path(output) if output else project_path(checkpoint).parent
    save_evaluation(destination, split, metrics, records, probabilities.tolist())
    print(json.dumps({k: metrics[k] for k in ('split', 'samples', 'accuracy', 'macro_f1', 'uar')}, indent=2), flush=True)
    return metrics


def train(cfg, resume=None):
    seed_all(cfg['seed'], cfg['train']['torch_threads'])
    cache = load_cache(cfg)
    output = project_path(cfg['train']['output_dir'])
    if not resume and any((output / name).exists() for name in ('best.pt', 'last.pt', 'history.jsonl')):
        raise FileExistsError(f'Existing run in {output}; use --resume or change train.output_dir.')
    device = device_for(cfg['train']['device'])
    train_ids, val_ids = (indices_for(cache['rows'], s) for s in ('train', 'val'))
    mean, scale = fit_standardizer(cache['features'], train_ids, cfg['features']['standardize'])
    head = Head(cfg['model'], mean, scale).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train']['epochs'], eta_min=cfg['train']['lr'] * .01)
    start, best, anchor, bad = 0, -1., -1., 0
    if resume:
        head_saved, saved = load_head(resume)
        expected, actual = copy.deepcopy(cfg), copy.deepcopy(saved['config'])
        for x in (expected, actual):
            for key in ('device', 'num_workers', 'torch_threads', 'tensorboard'):
                x['train'].pop(key, None)
        if expected != actual or cache['identity'] != saved['feature_identity']:
            raise ValueError('Resume requires the original configuration and feature cache.')
        if not (output / 'best.pt').exists():
            raise FileNotFoundError('Keep best.pt in the original output_dir when resuming.')
        head.load_state_dict(head_saved.state_dict(), strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, anchor, bad = saved['epoch'] + 1, saved['best_macro_f1'], saved['patience_anchor'], saved['bad_epochs']
        restore_rng(saved['rng'])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'config.json', cfg)
    save_json(output / 'labels.json', LABEL_MAP)
    logger = logging.getLogger('emotion2vec-course')
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(output / 'train.log'), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    x_train, y_train = cache['features'][train_ids].to(device), cache['labels'][train_ids].to(device)
    x_val, y_val = cache['features'][val_ids].to(device), cache['labels'][val_ids].to(device)
    logger.info('Frozen encoder=%s head=%s trainable_parameters=%d train=%d val=%d',
                REPO, cfg['model']['head'], sum(p.numel() for p in head.parameters()), len(train_ids), len(val_ids))
    experiment = ExperimentLog(output, cfg, enabled=cfg['train'].get('tensorboard', True),
                               resume_epoch=start + 1 if resume else None)
    started = time.monotonic()
    try:
        for epoch in range(start, cfg['train']['epochs']):
            if bad >= cfg['train']['patience']:
                break
            head.train()
            order = torch.randperm(len(train_ids), device=device)
            loss_sum, correct = 0., 0
            for batch_ids in order.split(cfg['train']['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                logits = head(x_train[batch_ids])
                loss = nn.functional.cross_entropy(logits, y_train[batch_ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite classification loss')
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), cfg['train']['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_ids)
                correct += int((logits.detach().argmax(-1) == y_train[batch_ids]).sum())
            val, _ = score_head(head, x_val, y_val)
            score = val['macro_f1']
            improved = score > best
            best = max(best, score)
            if score > anchor + cfg['train']['min_delta']:
                anchor, bad = score, 0
            else:
                bad += 1
            record = {'epoch': epoch + 1, 'train_loss': loss_sum / len(train_ids),
                      'train_accuracy': correct / len(train_ids), 'val_loss': val['loss'],
                      'val_accuracy': val['accuracy'], 'val_macro_f1': score, 'val_uar': val['uar'],
                      'best_macro_f1': best, 'bad_epochs': bad, 'lr': optimizer.param_groups[0]['lr'],
                      'elapsed_seconds': time.monotonic() - started}
            scheduler.step()
            state = pack(head, cfg, cache, epoch=epoch, best_macro_f1=best, patience_anchor=anchor,
                         bad_epochs=bad, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state())
            if improved:
                atomic_save(output / 'best.pt', state)
            atomic_save(output / 'last.pt', state)
            with (output / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            logger.info(json.dumps(record))
            experiment.epoch(record)
    except BaseException:
        experiment.close()
        logger.info('Training stopped; last.pt retains the last complete epoch.')
        raise
    logger.info('Finished; evaluate validation-selected best.pt on val and test.')
    try:
        for split in ('val', 'test'):
            evaluate(output / 'best.pt', split, device=str(device))
        best_epoch = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['epoch'] + 1
        experiment.final(output, best_epoch)
    finally:
        experiment.close()


def predict(checkpoint, audio, device='auto', encoder_dir=None):
    head, saved = load_head(checkpoint)
    cfg = saved['config']
    seed_all(cfg['seed'], cfg['train']['torch_threads'])
    device = device_for(device)
    encoder = FrozenEncoder(encoder_dir or cfg['model']['encoder_dir'], device, saved['encoder'])
    feature = encoder.audio(audio, cfg['data'])
    head.to(device).eval()
    with torch.no_grad():
        logits = head(feature[None, :].to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite single-file prediction')
        probabilities = logits.softmax(-1)[0].cpu().tolist()
    return {'audio': str(project_path(audio)), 'prediction': LABELS[int(np.argmax(probabilities))],
            'probabilities': dict(zip(LABELS, probabilities)), 'encoder': REPO,
            'head': cfg['model']['head'], 'max_seconds': cfg['data']['max_seconds']}


def smoke(cfg):
    seed_all(cfg['seed'], cfg['train']['torch_threads'])
    rows = read_manifest(cfg['data']['manifest'])
    chosen = [r for r in rows if r['split'] == 'train'][:2]
    device = device_for(cfg['train']['device'])
    encoder = FrozenEncoder(cfg['model']['encoder_dir'], device)
    vectors = torch.stack([encoder.audio(r['path'], cfg['data']) for r in chosen]).clone()
    # Clone outside inference_mode so head backward may save the input tensor.
    mean, scale = fit_standardizer(vectors, torch.arange(len(chosen)), cfg['features']['standardize'])
    head = Head(cfg['model'], mean, scale).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    before = [p.detach().clone() for p in head.parameters()]
    logits = head(vectors.to(device))
    loss = nn.functional.cross_entropy(logits, torch.tensor([int(r['label']) for r in chosen], device=device))
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
        raise AssertionError('Head backward produced invalid gradients')
    optimizer.step()
    if not any(not torch.equal(p, old) for p, old in zip(head.parameters(), before)):
        raise AssertionError('Head did not update')
    if encoder.model.training or any(p.requires_grad or p.grad is not None for p in encoder.model.parameters()):
        raise AssertionError('Encoder was not completely frozen')
    identity = cache_identity(cfg, rows, encoder.info)
    path = project_path('outputs/emotion2vec_smoke/smoke.pt')
    atomic_save(path, pack(head, cfg, {'identity': identity}))
    head.eval()
    with torch.no_grad():
        expected = head(vectors.to(device)).cpu()
    loaded, _ = load_head(path)
    with torch.no_grad():
        actual = loaded.to(device).eval()(vectors.to(device)).cpu()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    # Re-load encoder+head through the actual CLI prediction implementation.
    single = predict(path, chosen[0]['path'], str(device))
    torch.testing.assert_close(torch.tensor(list(single['probabilities'].values())), expected[0].softmax(-1), rtol=1e-4, atol=1e-5)
    report = {'status': 'passed', 'encoder': encoder.info, 'batch_size': len(chosen),
              'feature_shape': list(vectors.shape), 'loss': float(loss), 'encoder_frozen': True,
              'head_backward': True, 'head_updated': True, 'reload_equal': True,
              'single_file_prediction_matches': True, 'prediction': single}
    save_json(path.parent / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'encoder'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Frozen emotion2vec_base features + CREMA-D six-class Linear/MLP head')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'check-checkpoint', 'extract', 'train', 'smoke', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/emotion2vec.yaml')
        if name == 'prepare':
            p.add_argument('--force', action='store_true')
            p.add_argument('--skip-invalid', action='store_true')
        if name == 'extract':
            p.add_argument('--force', action='store_true')
        if name == 'train':
            p.add_argument('--resume')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--output')
    p.add_argument('--cache')
    p.add_argument('--device', default='auto')
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--encoder-dir')
    p.add_argument('--device', default='auto')
    p.add_argument('--output')
    p = sub.add_parser('export-tensorboard', help='Import an existing run without retraining')
    p.add_argument('--run-dir', default='outputs/cremad_emotion2vec_linear')
    args = parser.parse_args()
    try:
        if args.command in ('prepare', 'check-checkpoint', 'extract', 'train', 'smoke', 'run'):
            cfg = load_config(args.config)
            if cfg['model'].get('backend') != 'emotion2vec':
                raise ValueError('This entry point requires configs/emotion2vec.yaml; legacy Vesper: python -m ser.vesper')
        if args.command == 'prepare':
            prepare(cfg, args.force, args.skip_invalid)
        elif args.command == 'check-checkpoint':
            seed_all(cfg['seed'], cfg['train']['torch_threads'])
            encoder = FrozenEncoder(cfg['model']['encoder_dir'], device_for(cfg['train']['device']))
            print(json.dumps(encoder.info, indent=2))
        elif args.command == 'extract':
            extract(cfg, args.force)
        elif args.command == 'train':
            train(cfg, args.resume)
        elif args.command == 'smoke':
            smoke(cfg)
        elif args.command == 'run':
            extract(cfg)
            train(cfg)
        elif args.command == 'evaluate':
            evaluate(args.checkpoint, args.split, args.output, args.device, args.cache)
        elif args.command == 'predict':
            result = predict(args.checkpoint, args.audio, args.device, args.encoder_dir)
            if args.output:
                save_json(project_path(args.output), result)
            print(json.dumps(result, indent=2))
        elif args.command == 'export-tensorboard':
            export_history(args.run_dir)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
