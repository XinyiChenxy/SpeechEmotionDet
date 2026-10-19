import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
import torch

from .common import LABELS, save_json


def scores(targets, predictions):
    labels = list(range(6))
    return {'accuracy': float(accuracy_score(targets, predictions)),
            'macro_f1': float(f1_score(targets, predictions, labels=labels, average='macro', zero_division=0)),
            'uar': float(recall_score(targets, predictions, labels=labels, average='macro', zero_division=0)),
            'per_class': classification_report(targets, predictions, labels=labels, target_names=LABELS,
                                                output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(targets, predictions, labels=labels).tolist()}


@torch.no_grad()
def infer_loader(model, loader, device, amp=False):
    model.eval()
    targets, probabilities, rows = [], [], []
    total_loss = 0.0
    for batch in loader:
        waveform = batch['waveform'].to(device)
        mask = batch['padding_mask'].to(device)
        y = batch['label'].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            logits = model(waveform, mask)
            loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation output')
        targets.extend(y.cpu().tolist())
        probabilities.extend(logits.float().softmax(-1).cpu().tolist())
        rows.extend(batch['rows'])
        total_loss += float(loss) * len(y)
    if not targets:
        raise ValueError('Empty evaluation split')
    result = scores(targets, np.argmax(probabilities, axis=1))
    result['loss'] = total_loss / len(targets)
    result['samples'] = len(targets)
    return result, rows, probabilities


def save_evaluation(output, split, result, rows, probabilities):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / f'{split}_metrics.json', result)
    with (output / f'{split}_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'actor_id', 'true_emotion', 'predicted_emotion'] + [f'p_{x}' for x in LABELS])
        for row, probability in zip(rows, probabilities):
            writer.writerow([row['path'], row['actor_id'], row['emotion'], LABELS[int(np.argmax(probability))]] + probability)
    cm = np.asarray(result['confusion_matrix'])
    with (output / f'{split}_confusion_matrix.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true / predicted'] + LABELS)
        writer.writerows([[label] + row.tolist() for label, row in zip(LABELS, cm)])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted emotion', ylabel='Intended emotion', title=f'CREMA-D {split}')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output / f'{split}_confusion_matrix.png', dpi=160)
    plt.close(fig)
