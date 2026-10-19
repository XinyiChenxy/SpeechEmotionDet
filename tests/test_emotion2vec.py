"""Checks specific to the frozen-feature workflow, using no fake encoder scores."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from ser.audio import preprocess
from ser.common import LABELS, LABEL_MAP, atomic_save, load_config
from ser.emotion2vec import (Head, cache_identity, fit_standardizer, load_head,
                            score_head, train, validate_cache)


class FrozenFeatureTests(unittest.TestCase):
    def test_standardizer_uses_only_train_and_handles_constant_dimensions(self):
        features = torch.tensor([[1., 2.], [3., 2.], [99999., -99999.]])
        mean, scale = fit_standardizer(features, torch.tensor([0, 1]))
        torch.testing.assert_close(mean, torch.tensor([2., 2.]))
        torch.testing.assert_close(scale, torch.tensor([1., 1e-6]))
        self.assertTrue(torch.isfinite((features - mean) / scale).all())

    def test_full_utterance_policy(self):
        cfg = load_config('configs/emotion2vec.yaml')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'four_seconds.wav'
            sf.write(path, np.sin(np.arange(64000) * .1), 16000)
            waveform = preprocess(path, cfg['data'])
            self.assertEqual(waveform.numel(), 64000)
            self.assertAlmostEqual(float(waveform.mean()), 0, places=5)

    def test_linear_and_mlp_reload_probabilities_and_labels(self):
        cfg = load_config('configs/emotion2vec.yaml')
        for kind in ('linear', 'mlp'):
            config = copy.deepcopy(cfg)
            config['model']['head'] = kind
            head = Head(config['model'], torch.zeros(768), torch.ones(768))
            features = torch.randn(6, 768)
            loss = torch.nn.functional.cross_entropy(head(features), torch.arange(6))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            head.eval()
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'head.pt'
                atomic_save(path, {'format': 'emotion2vec-cremad-course-v1', 'labels': LABELS,
                                   'label_mapping': LABEL_MAP, 'config': config, 'head': head.state_dict()})
                restored, _ = load_head(path)
                restored.eval()
                torch.testing.assert_close(head(features), restored(features), rtol=0, atol=0)
                result, probabilities = score_head(restored, features, torch.arange(6))
                self.assertEqual(result['samples'], 6)
                np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-6)

    def test_cache_rejects_changed_labels_and_changed_audio(self):
        cfg = load_config('configs/emotion2vec.yaml')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'manifest.csv'
            manifest.write_text('fixture manifest')
            audio = root / 'fixture.wav'
            audio.write_bytes(b'fixture source bytes')
            cfg['data']['manifest'] = str(manifest)
            rows = [{'path': str(audio), 'label': '0'}]
            identity = cache_identity(cfg, rows, {'fixture': True})
            cache = {'format': 'emotion2vec-features-v1', 'identity': identity,
                     'rows': rows, 'features': torch.zeros(1, 768), 'labels': torch.tensor([0])}
            validate_cache(cache, cfg, rows)
            cache['labels'][0] = 1
            with self.assertRaisesRegex(ValueError, 'labels differ'):
                validate_cache(cache, cfg, rows)
            cache['labels'][0] = 0
            audio.write_bytes(b'changed fixture source bytes')
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate_cache(cache, cfg, rows)

    def test_resume_matches_uninterrupted_head_training(self):
        # Pure head training on hand-authored fixture features. Artifacts remain
        # inside a temporary directory; this is not a second SER experiment.
        cfg = load_config('configs/emotion2vec.yaml')
        cfg['train'].update(epochs=4, patience=10, batch_size=3, device='cpu', torch_threads=2)
        cfg['model']['head'] = 'mlp'  # Exercise dropout RNG restoration too.
        features = torch.sin(torch.arange(18 * 768).float().reshape(18, 768))
        cache = {'features': features, 'labels': torch.arange(18) % 6,
                 'rows': [{'split': split} for split in ('train', 'val', 'test') for _ in range(6)],
                 'identity': {'encoder': {'fixture': True}, 'preprocessing': {}}}

        class PlannedInterruption(Exception):
            pass

        with tempfile.TemporaryDirectory() as temp:
            cfg['train']['output_dir'] = str(Path(temp) / 'complete')
            with patch('ser.emotion2vec.load_cache', return_value=cache), patch('ser.emotion2vec.evaluate'):
                train(cfg)
            _, expected = load_head(Path(cfg['train']['output_dir']) / 'last.pt')
            cfg['train']['output_dir'] = str(Path(temp) / 'resumed')

            def interrupt_after_epoch(path, data):
                atomic_save(path, data)
                if Path(path).name == 'last.pt' and data['epoch'] == 1:
                    raise PlannedInterruption()

            with patch('ser.emotion2vec.load_cache', return_value=cache), patch('ser.emotion2vec.evaluate'):
                with patch('ser.emotion2vec.atomic_save', side_effect=interrupt_after_epoch):
                    with self.assertRaises(PlannedInterruption):
                        train(cfg)
                train(cfg, str(Path(cfg['train']['output_dir']) / 'last.pt'))
            _, actual = load_head(Path(cfg['train']['output_dir']) / 'last.pt')
            for key, value in expected['head'].items():
                torch.testing.assert_close(value, actual['head'][key], rtol=0, atol=0)
            self.assertEqual(expected['best_macro_f1'], actual['best_macro_f1'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
