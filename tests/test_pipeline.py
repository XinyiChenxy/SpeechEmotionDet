"""Offline correctness checks. No random encoder is trained or reported as Vesper."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from torch import nn

from ser.audio import collate, load_audio, preprocess
from ser.common import CODES, LABELS, load_config
from ser.data import inspect_manifest, prepare, read_manifest
from ser.metrics import save_evaluation, scores
from ser.model import EmotionModel, PaddingSafeVesper, feature_lengths, load_pretrained


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_config('configs/cremad.yaml')['data']

    def tearDown(self):
        self.temp.cleanup()

    def test_stereo_resample_normalize_and_padding(self):
        rate = 22050
        t = np.arange(rate) / rate
        signal = np.sin(2 * np.pi * 220 * t).astype('float32') * .2
        path = self.root / 'stereo.wav'
        sf.write(path, np.stack([signal, signal * .5], axis=1), rate, subtype='FLOAT')
        decoded, _ = load_audio(path)
        self.assertEqual(decoded.shape, (16000,))
        x = preprocess(path, self.cfg)
        self.assertAlmostEqual(float(x.mean()), 0, places=5)
        # layer_norm eps=1e-5 makes variance slightly below one at this RMS.
        self.assertAlmostEqual(float(x.var(unbiased=False)), 1, delta=.002)
        batch = collate([(x, 0, {}), (x[:8000], 1, {})])
        self.assertEqual(batch['padding_mask'][1].sum(), 8000)
        self.assertEqual(batch['padding_mask'][0].sum(), 0)
        self.assertEqual(float(batch['waveform'][1, 8000:].abs().sum()), 0)
        torch.testing.assert_close(batch['waveform'][0], x)

    def test_crop_is_consistent_and_eval_ignores_augmentation(self):
        path = self.root / 'long.wav'
        sf.write(path, np.sin(np.arange(64000) * .1), 16000)
        cfg = dict(self.cfg, augment=True)
        first = preprocess(path, cfg)
        second = preprocess(path, cfg, augment=False)
        self.assertEqual(first.numel(), 48000)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_lfs_corrupt_and_short_rejected(self):
        path = self.root / 'pointer.wav'
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n')
        with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
            load_audio(path)
        path = self.root / 'corrupt.wav'
        path.write_bytes(b'not audio')
        with self.assertRaises(Exception):
            load_audio(path)
        path = self.root / 'short.wav'
        sf.write(path, np.zeros(100), 16000)
        with self.assertRaisesRegex(ValueError, 'too short'):
            preprocess(path, self.cfg)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_mp3_decode(self):
        path = self.root / 'input.wav'
        sf.write(path, np.sin(np.arange(16000) * .1) * .2, 16000)
        mp3 = self.root / 'input.mp3'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), str(mp3)], check=True)
        audio, _ = load_audio(mp3)
        self.assertEqual(audio.numel(), 16000)
        self.assertTrue(torch.isfinite(preprocess(mp3, self.cfg)).all())


class SplitTests(unittest.TestCase):
    def test_actor_splits_coverage_determinism_and_x_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for actor in range(1001, 1013):
                for code in CODES:
                    suffix = 'X' if actor == 1001 and code == 'SAD' else 'XX'
                    sf.write(root / f'{actor}_ITH_{code}_{suffix}.wav', np.sin(np.arange(800) * .1), 16000)
            cfg = load_config('configs/cremad.yaml')
            cfg['data'].update(root=str(root), manifest=str(root / 'manifest.csv'))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(cfg)
            rows = read_manifest(cfg['data']['manifest'])
            summary = inspect_manifest(rows)
            self.assertEqual(sum(s['samples'] for s in summary.values()), 72)
            first = Path(cfg['data']['manifest']).read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(cfg, force=True)
            self.assertEqual(first, Path(cfg['data']['manifest']).read_bytes())
            audit = json.loads((root / 'audio_audit.json').read_text())
            self.assertEqual(len(audit['nonstandard_intensity_kept']), 1)
            # Deliberately leak one actor while leaving other classes present.
            broken = copy.deepcopy(rows)
            actor = next(r['actor_id'] for r in broken if r['split'] == 'train')
            next(r for r in broken if r['actor_id'] == actor)['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'Actor leakage'):
                inspect_manifest(broken)


class PaddingTests(unittest.TestCase):
    def test_exact_cnn_lengths_and_padding_mask(self):
        for n in [400, 719, 720, 16000, 16001, 47999, 48000]:
            x = torch.zeros(1, 1, n)
            # Functional convolutions with constant kernels test actual output
            # sizes without constructing or initializing a Vesper encoder.
            for k, s in [(10, 5)] + [(3, 2)] * 4 + [(2, 2)] * 2:
                x = torch.nn.functional.conv1d(x, torch.ones(1, 1, k), stride=s)
            self.assertEqual(int(feature_lengths(torch.tensor(n))), x.shape[-1])
        dummy = PaddingSafeVesper.__new__(PaddingSafeVesper)
        nn.Module.__init__(dummy)
        mask = torch.arange(48000)[None, :] >= torch.tensor([16000, 48000])[:, None]
        dummy.forward_padding_mask(torch.zeros(2, 149, 1), mask)
        self.assertEqual((~dummy.padding_mask).sum(1).tolist(), [49, 149])

    def test_biased_projector_padding_invariance_and_backward(self):
        class FeatureFixture(nn.Module):
            """Hand-authored tensor fixture, not an encoder/model substitute."""
            def forward(self, waveform, mask, ret_layer_results):
                self.mask = mask
                x = waveform[..., None]
                return x, [(x, None), (x * 2, None)]

            def get_padding_mask(self):
                return self.mask

        model = EmotionModel.__new__(EmotionModel)
        nn.Module.__init__(model)
        model.vesper = FeatureFixture()
        model.freeze_upstream = True
        model.weights = nn.Parameter(torch.zeros(2))
        model.projector = nn.Linear(1, 1)
        model.classifier = nn.Linear(1, 6)
        with torch.no_grad():
            model.projector.weight.fill_(1)
            model.projector.bias.fill_(5)
            model.classifier.weight.copy_(torch.arange(1., 7.)[:, None])
            model.classifier.bias.zero_()
        x = torch.tensor([[1., 2., 3.]])
        padded = torch.tensor([[1., 2., 3., 0., 0., 0.]])
        mask = torch.tensor([[False, False, False, True, True, True]])
        expected, actual = model(x), model(padded, mask)
        torch.testing.assert_close(expected, actual)
        torch.nn.functional.cross_entropy(actual, torch.tensor([0])).backward()
        self.assertTrue(torch.isfinite(model.weights.grad).all())
        self.assertGreater(float(model.projector.weight.grad.abs().sum()), 0)


class CheckpointAndMetricsTests(unittest.TestCase):
    def test_missing_and_wavlm_checkpoints_rejected(self):
        cfg = load_config('configs/cremad.yaml')['model']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'missing.pt'
            with self.assertRaisesRegex(FileNotFoundError, 'Genuine pretrained Vesper'):
                load_pretrained(cfg, path)
            torch.save({'model': {'encoder.layers.0.fc1.weight': torch.ones(2048)}}, path)
            with self.assertRaisesRegex(ValueError, 'WavLM state_dict'):
                load_pretrained(cfg, path)
            torch.save({'model': {'vesper.encoder.layers.0.fc1.weight': torch.ones(2048)}}, path)
            with self.assertRaisesRegex(ValueError, 'do not match Vesper-4'):
                load_pretrained(cfg, path)

    def test_metrics_and_artifact_schema(self):
        targets = [0, 1, 2, 3, 4, 5]
        result = scores(targets, targets)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['uar'], 1)
        with tempfile.TemporaryDirectory() as temp:
            rows = [{'path': str(i), 'actor_id': '1001', 'emotion': LABELS[i]} for i in targets]
            save_evaluation(temp, 'fixture', result, rows, np.eye(6).tolist())
            self.assertEqual(len(list(Path(temp).iterdir())), 4)
        # This fixture never writes fake results into project outputs.


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
