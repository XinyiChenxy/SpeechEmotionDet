import copy
import json
import logging
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .audio import collate, preprocess
from .common import (LABELS, LABEL_MAP, atomic_save, device_for, project_path, restore_rng,
                     rng_state, save_json, seed_all, sha256)
from .data import EmotionDataset, read_manifest
from .metrics import infer_loader, save_evaluation
from .model import load_pretrained, load_trained


def loader_for(rows, cfg, split, device, shuffle=False):
    return DataLoader(EmotionDataset(rows, cfg['data'], split),
                      batch_size=cfg['train']['batch_size'], shuffle=shuffle,
                      num_workers=cfg['train']['num_workers'], collate_fn=collate,
                      pin_memory=device.type == 'cuda', persistent_workers=False)


def get_logger(output):
    logger = logging.getLogger('ser')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / 'train.log'), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    return logger


def pack_checkpoint(model, cfg, manifest_hash, **extra):
    return {'format': 'vesper-cremad-course-v1', 'config': copy.deepcopy(cfg),
            'labels': LABELS, 'label_mapping': LABEL_MAP,
            'preprocessing': '16k mono; first max_seconds; unpadded utterance layer_norm; exact CNN mask',
            'manifest_sha256': manifest_hash, 'pretrained': model.provenance,
            'model': model.state_dict(), **extra}


def check_manifest(data, path):
    if sha256(path) != data['manifest_sha256']:
        raise ValueError('Manifest differs from training checkpoint. Restore the original split manifest.')


def check_resume(cfg, previous):
    for field in ('seed', 'data'):
        if cfg[field] != previous[field]:
            raise ValueError(f'Resume requires identical {field} configuration.')
    for section, ignored in [('model', {'checkpoint'}), ('train', {'device', 'num_workers', 'torch_threads'})]:
        now = {k: v for k, v in cfg[section].items() if k not in ignored}
        before = {k: v for k, v in previous[section].items() if k not in ignored}
        if now != before:
            raise ValueError(f'Resume requires identical {section} settings (including epochs/output_dir).')


def train(cfg, checkpoint=None, resume=None):
    cfg = copy.deepcopy(cfg)
    if checkpoint:
        cfg['model']['checkpoint'] = str(project_path(checkpoint))
    seed_all(cfg['seed'], cfg['train'].get('torch_threads', 4))
    device = device_for(cfg['train']['device'])
    # Fail before creating any apparent training results if weights are absent.
    if resume:
        model, previous = load_trained(resume)
        check_resume(cfg, previous['config'])
    else:
        model = load_pretrained(cfg['model'])
        previous = None
    manifest = project_path(cfg['data']['manifest'])
    rows = read_manifest(manifest)
    manifest_hash = sha256(manifest)
    if previous:
        check_manifest(previous, manifest)
    output = project_path(cfg['train']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if previous is None and any((output / name).exists() for name in ('best.pt', 'last.pt', 'history.jsonl')):
        raise FileExistsError(f'Existing run in {output}; use --resume or a new output_dir.')
    if previous and not (output / 'best.pt').is_file():
        raise FileNotFoundError('Resume requires best.pt beside the run logs, so the prior best is preserved.')
    logger = get_logger(output)
    save_json(output / 'config.json', cfg)
    save_json(output / 'labels.json', LABEL_MAP)
    train_loader = loader_for(rows, cfg, 'train', device, shuffle=True)
    val_loader = loader_for(rows, cfg, 'val', device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    options = cfg['train']
    if options['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=options['lr'], weight_decay=options['weight_decay'],
                                    momentum=options.get('momentum', 0.9))
    elif options['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=options['lr'], weight_decay=options['weight_decay'])
    else:
        raise ValueError('optimizer must be sgd or adamw')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'], eta_min=options['lr'] * .01)
    amp = bool(options.get('amp', True)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start, best, anchor, bad_epochs = 0, -1.0, -1.0, 0
    if previous:
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        start, best = previous['epoch'] + 1, previous['best_macro_f1']
        anchor, bad_epochs = previous['patience_anchor'], previous['bad_epochs']
        restore_rng(previous['rng'])
    logger.info('Device=%s pretrained=%s trainable_parameters=%d', device, model.provenance, sum(p.numel() for p in params))
    try:
        for epoch in range(start, options['epochs']):
            if bad_epochs >= options['patience']:
                break
            begin = time.monotonic()
            model.train()
            loss_sum, count = 0.0, 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{options["epochs"]}', mininterval=2):
                optimizer.zero_grad(set_to_none=True)
                x, mask, y = batch['waveform'].to(device), batch['padding_mask'].to(device), batch['label'].to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(x, mask)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, options.get('grad_clip', 5.0), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                count += len(y)
                loss_sum += float(loss.detach()) * len(y)
            metrics, _, _ = infer_loader(model, val_loader, device, amp=amp)
            score = metrics['macro_f1']
            is_best = score > best
            if is_best:
                best = score
            if score > anchor + options.get('min_delta', 0.0001):
                anchor, bad_epochs = score, 0
            else:
                bad_epochs += 1
            record = {'epoch': epoch + 1, 'train_loss': loss_sum / count,
                      'val_loss': metrics['loss'], 'val_accuracy': metrics['accuracy'],
                      'val_macro_f1': score, 'val_uar': metrics['uar'], 'best_macro_f1': best,
                      'bad_epochs': bad_epochs, 'lr': optimizer.param_groups[0]['lr'],
                      'seconds': time.monotonic() - begin}
            scheduler.step()
            state = pack_checkpoint(model, cfg, manifest_hash, epoch=epoch, best_macro_f1=best,
                                    patience_anchor=anchor, bad_epochs=bad_epochs,
                                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                    scaler=scaler.state_dict(), rng=rng_state())
            if is_best:
                atomic_save(output / 'best.pt', state)
            atomic_save(output / 'last.pt', state)
            with (output / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            logger.info(json.dumps(record))
        logger.info('Training finished. Evaluating validation-selected best checkpoint on val/test.')
        for split in ('val', 'test'):
            evaluate(output / 'best.pt', split=split, device=str(device))
    except KeyboardInterrupt:
        logger.info('Interrupted; last.pt contains the last completed epoch. Resume with --resume %s', output / 'last.pt')
        raise


def evaluate(checkpoint, split='test', device='auto', manifest=None, output=None):
    model, saved = load_trained(checkpoint)
    cfg = saved['config']
    seed_all(cfg['seed'], cfg['train'].get('torch_threads', 4))
    device = device_for(device)
    manifest = project_path(manifest or cfg['data']['manifest'])
    check_manifest(saved, manifest)
    rows = read_manifest(manifest)
    loader = loader_for(rows, cfg, split, device)
    model.to(device)
    metrics, records, probabilities = infer_loader(model, loader, device, amp=False)
    metrics.update(checkpoint=str(project_path(checkpoint)), checkpoint_sha256=sha256(project_path(checkpoint)),
                   split=split, manifest_sha256=saved['manifest_sha256'])
    save_evaluation(project_path(output) if output else project_path(checkpoint).parent, split, metrics, records, probabilities)
    print(json.dumps({k: metrics[k] for k in ('split', 'samples', 'accuracy', 'macro_f1', 'uar')}, indent=2))
    return metrics


@torch.no_grad()
def predict(checkpoint, audio, device='auto'):
    model, data = load_trained(checkpoint)
    seed_all(data['config']['seed'], data['config']['train'].get('torch_threads', 4))
    device = device_for(device)
    model.to(device).eval()
    x = preprocess(project_path(audio), data['config']['data']).unsqueeze(0).to(device)
    logits = model(x)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite prediction')
    probabilities = logits.float().softmax(-1)[0].cpu().tolist()
    return {'audio': str(project_path(audio)), 'prediction': LABELS[max(range(6), key=probabilities.__getitem__)],
            'probabilities': dict(zip(LABELS, probabilities)),
            'processed_seconds': x.shape[1] / data['config']['data']['sample_rate']}


def smoke(cfg, checkpoint=None):
    """Real data and genuine pretrained weights only; creates no formal result scores."""
    seed_all(cfg['seed'], cfg['train'].get('torch_threads', 4))
    model = load_pretrained(cfg['model'], checkpoint)
    rows = read_manifest(cfg['data']['manifest'])
    device = device_for(cfg['train']['device'])
    samples = EmotionDataset(rows, cfg['data'], 'train')
    batch = collate([samples[i] for i in range(min(2, len(samples)))])
    model.to(device).train()
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    logits = model(batch['waveform'].to(device), batch['padding_mask'].to(device))
    loss = torch.nn.functional.cross_entropy(logits, batch['label'].to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite smoke loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError('Missing or non-finite gradients')
    optimizer.step()
    if torch.equal(before, model.classifier.weight):
        raise AssertionError('Classifier did not update')
    model.eval()
    with torch.no_grad():
        expected = model(batch['waveform'].to(device), batch['padding_mask'].to(device)).cpu()
    output = project_path('outputs/smoke')
    path = output / 'smoke.pt'
    atomic_save(path, pack_checkpoint(model, cfg, sha256(project_path(cfg['data']['manifest']))))
    loaded, _ = load_trained(path)
    loaded.to(device).eval()
    with torch.no_grad():
        actual = loaded(batch['waveform'].to(device), batch['padding_mask'].to(device)).cpu()
        single = model(batch['waveform'][:1, :int((~batch['padding_mask'][0]).sum())].to(device)).cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(expected[:1], single, rtol=1e-4, atol=1e-4)
    result = {'status': 'passed', 'batch_size': len(batch['rows']), 'loss': float(loss),
              'backward': True, 'classifier_updated': True, 'reload_equal': True,
              'padding_invariance': True, 'pretrained': model.provenance,
              'prediction': predict(path, batch['rows'][0]['path'], str(device))}
    save_json(output / 'report.json', result)
    print(json.dumps(result, indent=2))
