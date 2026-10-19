import argparse
import json
import sys

from .common import load_config, save_json


def main():
    parser = argparse.ArgumentParser(description='基于官方 Vesper 实现的 CREMA-D 情绪分类课程项目')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'check-checkpoint', 'train', 'smoke'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/cremad.yaml')
        if name in ('check-checkpoint', 'train', 'smoke'):
            p.add_argument('--checkpoint', help='Explicit genuine Vesper pretrained checkpoint path')
        if name == 'train':
            p.add_argument('--resume', help='Course last.pt; resumes at the next complete epoch')
        if name == 'prepare':
            p.add_argument('--force', action='store_true', help='Explicitly regenerate existing manifest')
            p.add_argument('--skip-invalid', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True, help='Trained course best.pt')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--manifest')
    p.add_argument('--output')
    p.add_argument('--device', default='auto')
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True, help='Trained course best.pt')
    p.add_argument('--audio', required=True, help='WAV or MP3')
    p.add_argument('--device', default='auto')
    p.add_argument('--output', help='Optional JSON prediction output')
    args = parser.parse_args()
    try:
        if args.command in ('prepare', 'check-checkpoint', 'train', 'smoke'):
            cfg = load_config(args.config)
        if args.command == 'prepare':
            from .data import prepare
            prepare(cfg, force=args.force, skip_invalid=args.skip_invalid)
        elif args.command == 'check-checkpoint':
            from .model import load_pretrained
            model = load_pretrained(cfg['model'], args.checkpoint)
            print(json.dumps(model.provenance, indent=2))
        elif args.command == 'train':
            from .engine import train
            train(cfg, args.checkpoint, args.resume)
        elif args.command == 'smoke':
            from .engine import smoke
            smoke(cfg, args.checkpoint)
        elif args.command == 'evaluate':
            from .engine import evaluate
            evaluate(args.checkpoint, args.split, args.device, args.manifest, args.output)
        elif args.command == 'predict':
            from .engine import predict
            result = predict(args.checkpoint, args.audio, args.device)
            if args.output:
                from .common import project_path
                save_json(project_path(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
