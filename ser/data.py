from collections import Counter
import csv
from pathlib import Path
import random
import re

from torch.utils.data import Dataset

from .audio import load_audio, preprocess
from .common import CODES, LABELS, LABEL_MAP, project_path, save_json, sha256

FIELDS = ['path', 'actor_id', 'emotion_code', 'label', 'emotion', 'duration', 'split']
# The distributed corpus contains 1040_ITH_SAD_X.wav. Intensity does not
# determine the target; keep this valid SAD utterance without renaming it.
PATTERN = re.compile(r'^(\d{4})_[A-Z]{3}_(ANG|DIS|FEA|HAP|NEU|SAD)_(LO|MD|HI|XX|X)$')


def inspect_manifest(rows):
    groups = {}
    seen = set()
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid split: {row}')
        if row['emotion_code'] not in LABEL_MAP or int(row['label']) != LABEL_MAP[row['emotion_code']]['id']:
            raise ValueError(f'Invalid label mapping: {row}')
        if row['emotion'] != LABELS[int(row['label'])]:
            raise ValueError(f'Invalid emotion name: {row}')
        match = PATTERN.fullmatch(Path(row['path']).stem)
        if not match or match[1] != row['actor_id'] or match[2] != row['emotion_code']:
            raise ValueError(f'Filename/actor/label mismatch: {row}')
        if row['path'] in seen:
            raise ValueError(f'Duplicate manifest path: {row["path"]}')
        if float(row['duration']) < 0.025:
            raise ValueError(f'Invalid duration: {row}')
        seen.add(row['path'])
        actor = row['actor_id']
        previous = groups.setdefault(actor, row['split'])
        if previous != row['split']:
            raise ValueError(f'Actor leakage: {actor}')
    summary = {}
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        counts = Counter(r['emotion_code'] for r in subset)
        if set(counts) != set(CODES):
            raise ValueError(f'{split} does not cover all six classes: {dict(counts)}')
        summary[split] = {'samples': len(subset), 'actors': sorted({r['actor_id'] for r in subset}),
                          'class_counts': {c: counts[c] for c in CODES}}
    return summary


def read_manifest(path):
    path = project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Manifest missing: {path}. Run python -m ser prepare first.')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    inspect_manifest(rows)
    return rows


def prepare(cfg, force=False, skip_invalid=False):
    root = project_path(cfg['data']['root'])
    destination = project_path(cfg['data']['manifest'])
    if destination.exists() and not force:
        rows = read_manifest(destination)
        print(f'Keeping existing manifest {destination}: {inspect_manifest(rows)}')
        return
    candidates = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.wav', '.mp3'))
    if not candidates:
        raise FileNotFoundError(f'No WAV/MP3 in {root}. Place official AudioWAV here; see README.md.')
    # Prefer lossless WAV when the same utterance exists in both encodings.
    selected, duplicates = {}, []
    for p in sorted(candidates, key=lambda p: (p.suffix.lower() != '.wav', str(p))):
        if p.stem in selected:
            previous = selected[p.stem]
            if p.suffix.lower() == previous.suffix.lower():
                raise ValueError(f'Ambiguous duplicate utterance: {previous} and {p}')
            duplicates.append(str(p))
        else:
            selected[p.stem] = p
    rows, rejected = [], []
    for name, path in sorted(selected.items()):
        try:
            match = PATTERN.fullmatch(name)
            if not match:
                raise ValueError('Not an intended-emotion CREMA-D filename')
            audio, _ = load_audio(path, cfg['data']['sample_rate'])
            if audio.numel() < 400:
                raise ValueError('Too short for Vesper CNN')
            code = match[2]
            rows.append({'path': str(path.resolve()), 'actor_id': match[1], 'emotion_code': code,
                         'label': LABEL_MAP[code]['id'], 'emotion': LABEL_MAP[code]['emotion'],
                         'duration': round(audio.numel() / cfg['data']['sample_rate'], 6)})
        except Exception as e:
            rejected.append({'path': str(path), 'error': str(e)})
    audit = destination.with_name('audio_audit.json')
    save_json(audit, {'scanned': len(candidates), 'valid': len(rows), 'rejected': rejected,
                      'nonstandard_intensity_kept': [r['path'] for r in rows if Path(r['path']).stem.endswith('_X')],
                      'ignored_mp3_duplicates': duplicates})
    if rejected and not skip_invalid:
        raise ValueError(f'{len(rejected)} invalid audio files; see {audit}. Fix them or explicitly --skip-invalid.')
    actors = sorted({r['actor_id'] for r in rows})
    random.Random(cfg['seed']).shuffle(actors)
    ratio = cfg['data']['train_ratio']
    val_ratio = cfg['data']['val_ratio']
    if not (0 < ratio < 1 and 0 < val_ratio < 1 and ratio + val_ratio < 1):
        raise ValueError('Invalid split ratios')
    n_train, n_val = round(len(actors) * ratio), round(len(actors) * val_ratio)
    actor_split = {actor: ('train' if i < n_train else 'val' if i < n_train + n_val else 'test')
                   for i, actor in enumerate(actors)}
    for row in rows:
        row['split'] = actor_split[row['actor_id']]
    summary = inspect_manifest(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_json(destination.with_suffix('.summary.json'),
              {'seed': cfg['seed'], 'label_source': 'filename intended emotion, not crowd votes',
               'label_mapping': LABEL_MAP, 'manifest_sha256': sha256(destination),
               'splits': summary, 'actor_overlap': False, 'audio_audit': str(audit)})
    print(f'Manifest saved: {destination}')
    for name, info in summary.items():
        print(name, info['samples'], 'samples', len(info['actors']), 'actors', info['class_counts'])


class EmotionDataset(Dataset):
    def __init__(self, rows, data_cfg, split):
        self.rows = [r for r in rows if r['split'] == split]
        self.data_cfg = data_cfg
        self.augment = split == 'train'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return preprocess(row['path'], self.data_cfg, augment=self.augment), int(row['label']), row
