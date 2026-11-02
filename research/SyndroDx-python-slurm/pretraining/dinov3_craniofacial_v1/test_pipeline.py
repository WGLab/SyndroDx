import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pipeline_supervisor as supervisor
import evaluate_encoder as evaluator
from runtime import save, digest


class PipelineTests(unittest.TestCase):
    def test_unknown_timestamp_does_not_kill_or_restart(self):
        self.assertEqual(supervisor.age(None), 0)

    def test_healthy_training_does_not_touch_processes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save(root / 'training/state.json', dict(state='training', updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            save(root / 'training/launcher.json', dict(state='running', started_at_utc='synthetic'))
            with patch.object(supervisor, 'ROOT', root), patch.object(supervisor, 'FOLDER', root / 'pipeline'), patch.object(supervisor.subprocess, 'run') as run:
                self.assertEqual(supervisor.training_monitor({}, set()), 'monitoring_training')
                run.assert_not_called()

    def test_full_evaluation_handoff_on_synthetic_features(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            folder = root / 'evaluation'
            folder.mkdir()
            rows, matrix, labels = [], [], {}
            for split in ['train', 'validation', 'test']:
                for cls, direction in [('A', [1., 0., .1]), ('B', [0., 1., .1])]:
                    group = split + cls
                    labels[group] = [cls]
                    for view in ['frontal', 'profile']:
                        rows.append(dict(group_id=group, split=split, view=view, pixel_sha256=group + view))
                        matrix.append(direction)
            path = folder / 'inputs.private.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            sha = digest(path)
            save(folder / 'input_plan.json', dict(world_size=2, inputs_sha256=sha))
            save(folder / 'labels.private.json', dict(state='synthetic_only', labels={'disease_or_disorder': labels}))
            save(folder / 'labels_status.json', dict(state='exploratory_metadata_labels_ready'))
            for slot in range(2):
                destination = folder / 'features' / f'slot_{slot:02d}'
                destination.mkdir(parents=True)
                indices = np.arange(slot, len(rows), 2)
                np.save(destination / 'indices.private.npy', indices)
                for kind in ['original_dinov3', 'craniofacial']:
                    feature_path = destination / (kind + '.private.npy')
                    np.save(feature_path, np.array(matrix, np.float32)[indices])
                    save(destination / (kind + '.json'), dict(state='complete', inputs_sha256=sha, features_sha256=digest(feature_path)))
            with patch.object(evaluator, 'ROOT', root):
                evaluator.evaluate()
            result = json.loads((folder / 'completion.json').read_text())
            self.assertEqual(result['state'], 'evaluation_complete')
            self.assertFalse(result['clinically_validated'])
            report = json.loads((folder / 'report.private.json').read_text())
            self.assertEqual(report['metadata_endpoints']['disease_or_disorder']['paired_comparison']['1']['delta_macro'], 0)


if __name__ == '__main__':
    unittest.main()
