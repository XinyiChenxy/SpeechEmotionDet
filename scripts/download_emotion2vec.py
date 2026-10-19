"""Download the exact user-requested base model, checking its published LFS SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = 'emotion2vec/emotion2vec_base'
REVISION = '0c9a3152734f9d7a7a05b4ee6bfb3c109d288664'
WEIGHT_SHA256 = '4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd'
WEIGHT_SIZE = 1125606009


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'checkpoints' / 'emotion2vec_base'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('README.md', 'config.yaml', 'configuration.json', 'emotion2vec_base.pt'):
        target = output / name
        if name == 'emotion2vec_base.pt' and target.exists() and digest(target) == WEIGHT_SHA256:
            print(f'Verified existing {target}', flush=True)
            continue
        if name != 'emotion2vec_base.pt' and target.exists():
            continue
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'
        temp = target.with_suffix(target.suffix + '.part')
        for attempt in range(3):
            try:
                print(f'Downloading {name} (attempt {attempt + 1})', flush=True)
                request = urllib.request.Request(url, headers={'User-Agent': 'emotion2vec-course-project'})
                h = hashlib.sha256()
                count, last = 0, time.monotonic()
                with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as f:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        count += len(chunk)
                        if time.monotonic() - last > 10:
                            print(f'{name}: {count / 1024**2:.0f} MiB', flush=True)
                            last = time.monotonic()
                if name == 'emotion2vec_base.pt' and (count != WEIGHT_SIZE or h.hexdigest() != WEIGHT_SHA256):
                    raise ValueError('Weight size/SHA256 does not match the published LFS object.')
                temp.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    provenance = {'repo': REPO, 'revision': REVISION, 'weight_sha256': WEIGHT_SHA256,
                  'weight_size': WEIGHT_SIZE, 'source': f'https://huggingface.co/{REPO}',
                  'sha256_verified': True}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Weights available: {output}', flush=True)


if __name__ == '__main__':
    main()
