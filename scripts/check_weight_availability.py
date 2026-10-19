"""Audit public author endpoints; never use credentials or guess weight URLs."""
import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = {
    'readme': 'https://raw.githubusercontent.com/HappyColor/Vesper/master/README.md',
    'releases': 'https://api.github.com/repos/HappyColor/Vesper/releases',
    'tags': 'https://api.github.com/repos/HappyColor/Vesper/tags',
    'weight_issue_2': 'https://api.github.com/repos/HappyColor/Vesper/issues/2',
    'weight_issue_2_comments': 'https://api.github.com/repos/HappyColor/Vesper/issues/2/comments',
    'weight_issue_3': 'https://api.github.com/repos/HappyColor/Vesper/issues/3',
    'weight_issue_3_comments': 'https://api.github.com/repos/HappyColor/Vesper/issues/3/comments',
}


def main():
    result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'endpoints': {}}
    for name, url in ENDPOINTS.items():
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Vesper-CREMA-D-course-project'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read().decode('utf-8')
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass
                result['endpoints'][name] = {'url': url, 'status': response.status, 'content': content}
        except Exception as error:
            result['endpoints'][name] = {'url': url, 'error': str(error)}
        print(name, result['endpoints'][name].get('status', 'failed'), flush=True)
    destination = ROOT / 'outputs' / 'weight_availability.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved {destination}; manually inspect author links before using any new checkpoint.')


if __name__ == '__main__':
    main()
