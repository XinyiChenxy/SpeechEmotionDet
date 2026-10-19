import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import create_app
from ser.common import LABELS, ROOT


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.info = {'best_epoch': 39, 'val_macro_f1': .74, 'encoder': 'emotion2vec/emotion2vec_base',
                     'head': 'linear', 'selection': 'validation Macro-F1', 'labels': LABELS}
        self.predictor = Mock()
        self.predictor.info = self.info
        self.predictor.predict_bytes.return_value = {
            'filename': 'fixture.wav', 'prediction': 'angry', 'probabilities': dict.fromkeys(LABELS, 1 / 6),
            'duration_seconds': 1., 'inference_seconds': .1, 'best_epoch': 39,
            'checkpoint_sha256': 'fixture', 'encoder': 'emotion2vec/emotion2vec_base', 'head': 'linear'}
        self.factory = Mock(return_value=self.predictor)
        self.client = TestClient(create_app(self.factory, lambda: self.info))

    def test_english_page_assets_metadata_and_docs(self):
        with self.client as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('<html lang="en">', response.text)
            self.assertNotRegex(response.text, '[\u4e00-\u9fff]')
            for asset in ('app.js', 'styles.css', 'favicon.svg', 'recorder.js', 'pcm-recorder.js'):
                self.assertEqual(client.get(f'/static/{asset}').status_code, 200)
            self.assertIn('Record audio', response.text)
            self.assertEqual(client.get('/api/model').json()['best_epoch'], 39)
            self.assertEqual(client.get('/health').json(), {'status': 'ok', 'model_loaded': False})
            self.assertIn('/api/predict', client.get('/openapi.json').json()['paths'])
            self.factory.assert_not_called()

    def test_upload_uses_cached_predictor_and_returns_six_probabilities(self):
        with self.client as client:
            for _ in range(2):
                response = client.post('/api/predict', files={'file': ('clip.wav', b'fixture bytes', 'audio/wav')})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()['probabilities']), set(LABELS))
            self.factory.assert_called_once()
            self.assertTrue(client.get('/health').json()['model_loaded'])
            self.predictor.predict_bytes.assert_called_with(b'fixture bytes', 'clip.wav')

    def test_invalid_files_are_rejected_before_loading_model(self):
        with self.client as client:
            invalid = client.post('/api/predict', files={'file': ('clip.txt', b'bytes')})
            self.assertEqual(invalid.status_code, 400)
            self.assertIn('WAV or MP3', invalid.json()['detail'])
            empty = client.post('/api/predict', files={'file': ('clip.wav', b'')})
            self.assertEqual(empty.status_code, 400)
            self.assertIn('empty', empty.json()['detail'])
            self.assertEqual(client.post('/api/predict').status_code, 422)
            oversized = client.post('/api/predict', content=b'', headers={'content-length': str(22 * 1024 * 1024)})
            self.assertEqual(oversized.status_code, 413)
            self.factory.assert_not_called()

    def test_decode_error_is_an_english_client_error(self):
        self.predictor.predict_bytes.side_effect = ValueError('The audio could not be decoded.')
        with self.client as client:
            response = client.post('/api/predict', files={'file': ('broken.mp3', b'broken')})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()['detail'], 'The audio could not be decoded.')

    def test_model_load_error_reports_unavailable(self):
        self.factory.side_effect = FileNotFoundError('private filesystem path')
        with self.client as client:
            response = client.post('/api/predict', files={'file': ('clip.wav', b'bytes')})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private filesystem path', response.text)

    def test_example_route_only_serves_allowlisted_audio(self):
        with self.client as client:
            examples = client.get('/api/examples').json()
            if examples:
                response = client.get(examples[0]['audio_url'])
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.content.startswith(b'RIFF'))
            self.assertEqual(client.get('/api/examples/not-a-sample/audio').status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
