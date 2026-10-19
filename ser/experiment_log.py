"""TensorBoard views of real epoch records and final saved evaluations."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import LABELS, project_path, save_json

SCALARS = {
    'train_loss': 'Loss/train', 'val_loss': 'Loss/val',
    'train_accuracy': 'Accuracy/train', 'val_accuracy': 'Accuracy/val',
    'val_macro_f1': 'MacroF1/val', 'val_uar': 'UAR/val',
    'lr': 'LearningRate', 'best_macro_f1': 'Best/val_macro_f1',
    'bad_epochs': 'EarlyStopping/bad_epochs',
}


class ExperimentLog:
    def __init__(self, run_dir, config, enabled=True, resume_epoch=None, origin='live training'):
        self.writer = None
        if not enabled:
            return
        from torch.utils.tensorboard import SummaryWriter
        self.directory = Path(run_dir) / 'tensorboard'
        self.writer = SummaryWriter(str(self.directory), purge_step=resume_epoch, flush_secs=10)
        self.writer.add_text('Experiment/config', '```json\n' + json.dumps(config, indent=2) + '\n```', 0)
        self.writer.add_text('Experiment/origin', origin, 0)
        self.writer.add_text('Experiment/labels', ', '.join(LABELS), 0)

    def epoch(self, record):
        if self.writer is None:
            return
        step = int(record['epoch'])  # Same one-based epoch as history.jsonl.
        for field, tag in SCALARS.items():
            if field in record:
                self.writer.add_scalar(tag, float(record[field]), step)
        self.writer.flush()

    def final(self, run_dir, best_epoch):
        if self.writer is None:
            return
        root = Path(run_dir)
        self.writer.add_scalar('Summary/best_epoch', best_epoch, best_epoch)
        for split in ('val', 'test'):
            path = root / f'{split}_metrics.json'
            if not path.exists():
                continue
            metrics = json.loads(path.read_text())
            for key in ('loss', 'accuracy', 'macro_f1', 'uar'):
                if key in metrics:
                    self.writer.add_scalar(f'Final/{split}/{key}', metrics[key], best_epoch)
            for label in LABELS:
                for key in ('precision', 'recall', 'f1-score', 'support'):
                    value = metrics.get('per_class', {}).get(label, {}).get(key)
                    if value is not None:
                        self.writer.add_scalar(f'Final/{split}/{label}/{key}', value, best_epoch)
            image = root / f'{split}_confusion_matrix.png'
            if image.exists():
                with Image.open(image) as source:
                    array = np.asarray(source.convert('RGB')).copy()
                self.writer.add_image(f'ConfusionMatrix/{split}', array, best_epoch, dataformats='HWC')
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


def export_history(run_dir):
    """Import existing results once, without rerunning training or changing checkpoints."""
    root = project_path(run_dir)
    history = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines() if line.strip()]
    if not history:
        raise ValueError('Cannot export an empty experiment history.')
    directory = root / 'tensorboard'
    if any(directory.glob('events.out.tfevents.*')):
        print(f'TensorBoard events already exist in {directory}; keeping them unchanged.')
        return directory
    config = json.loads((root / 'config.json').read_text())
    # Use the last completed occurrence of each epoch if a crash/restart left
    # duplicate records. Incomplete future records never enter these files.
    records = {int(row['epoch']): row for row in history}
    best = max(records.values(), key=lambda row: row['val_macro_f1'])
    log = ExperimentLog(root, config, origin='Imported from existing history.jsonl and final metrics; event wall times are import times.')
    try:
        for step in sorted(records):
            log.epoch(records[step])
        log.final(root, int(best['epoch']))
    finally:
        log.close()
    save_json(directory / 'import.json', {'source': str(root / 'history.jsonl'),
                                        'epochs': len(records), 'best_epoch': int(best['epoch']),
                                        'retrained': False})
    print(f'Imported {len(records)} epochs into {directory}')
    return directory
