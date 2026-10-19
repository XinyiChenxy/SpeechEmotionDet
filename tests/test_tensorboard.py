import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ser.common import LABELS
from ser.experiment_log import ExperimentLog, export_history


class TensorBoardTests(unittest.TestCase):
    def test_history_import_is_faithful_and_repeat_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history = [dict(epoch=1, train_loss=1.2, val_loss=1.3, val_accuracy=.4,
                            val_macro_f1=.35, val_uar=.4, lr=.001, best_macro_f1=.35, bad_epochs=0),
                       dict(epoch=2, train_loss=.9, val_loss=1.1, val_accuracy=.5,
                            val_macro_f1=.45, val_uar=.5, lr=.0009, best_macro_f1=.45, bad_epochs=0)]
            (root / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
            (root / 'config.json').write_text(json.dumps({'seed': 42}))
            metrics = {'accuracy': .5, 'macro_f1': .45, 'uar': .5, 'loss': 1.1,
                       'per_class': {label: {'precision': .5, 'recall': .5, 'f1-score': .5, 'support': 2}
                                     for label in LABELS}}
            for split in ('val', 'test'):
                (root / f'{split}_metrics.json').write_text(json.dumps(metrics))
                Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(root / f'{split}_confusion_matrix.png')
            directory = export_history(root)
            events = EventAccumulator(str(directory)).Reload()
            self.assertEqual([row.step for row in events.Scalars('Loss/train')], [1, 2])
            self.assertAlmostEqual(events.Scalars('MacroF1/val')[-1].value, .45, places=6)
            self.assertEqual(events.Scalars('Final/test/accuracy')[0].step, 2)
            self.assertIn('ConfusionMatrix/test', events.Tags()['images'])
            self.assertNotIn('Accuracy/train', events.Tags()['scalars'])  # Never invent historical values.
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            export_history(root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_live_resume_purges_stale_future_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            first = ExperimentLog(temp, {})
            for epoch in (1, 2, 3):
                first.epoch({'epoch': epoch, 'val_macro_f1': .1 * epoch})
            first.close()
            resumed = ExperimentLog(temp, {}, resume_epoch=2)
            resumed.epoch({'epoch': 2, 'val_macro_f1': .8, 'train_accuracy': .9})
            resumed.close()
            events = EventAccumulator(str(Path(temp) / 'tensorboard')).Reload()
            values = events.Scalars('MacroF1/val')
            self.assertEqual([row.step for row in values], [1, 2])
            self.assertAlmostEqual(values[-1].value, .8, places=6)
            self.assertAlmostEqual(events.Scalars('Accuracy/train')[0].value, .9, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
