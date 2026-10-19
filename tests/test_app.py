from pathlib import Path
import unittest
from unittest.mock import patch

from ser.app_service import BestModelPredictor, MAX_UPLOAD_BYTES, best_checkpoint_info


class AppInputTests(unittest.TestCase):
    def test_last_checkpoint_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'best.pt'):
            best_checkpoint_info(Path('last.pt'))

    def test_invalid_uploads_fail_before_model_inference(self):
        predictor = BestModelPredictor.__new__(BestModelPredictor)
        with self.assertRaisesRegex(ValueError, 'WAV or MP3'):
            predictor.predict_bytes(b'bytes', 'audio.txt')
        with self.assertRaisesRegex(ValueError, 'empty'):
            predictor.predict_bytes(b'', 'audio.wav')
        with self.assertRaisesRegex(ValueError, '20 MB'):
            predictor.predict_bytes(b'x' * (MAX_UPLOAD_BYTES + 1), 'audio.mp3')

    def test_corrupt_upload_cleanup_and_safe_filename(self):
        predictor = BestModelPredictor.__new__(BestModelPredictor)
        predictor.cfg = {'data': {}}
        paths = []

        def fail_decode(path, cfg, augment):
            paths.append(path)
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, 'input.wav')
            self.assertEqual(path.read_bytes(), b'broken audio')
            raise ValueError('decoder details')

        with patch('ser.app_service.preprocess', side_effect=fail_decode):
            with self.assertRaisesRegex(ValueError, 'could not be decoded'):
                predictor.predict_bytes(b'broken audio', '../../arbitrary.wav')
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].parent.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
