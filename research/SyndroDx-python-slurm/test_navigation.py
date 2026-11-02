"""Test code installation boundaries without reading any clinical data."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from catalog import CATALOG
import syndrodx


class NavigationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = io.StringIO()

    def tearDown(self):
        self.temp.cleanup()

    def install(self, **kwargs):
        with contextlib.redirect_stdout(self.output):
            return syndrodx.install(self.root, **kwargs)

    def test_preview_writes_nothing(self):
        self.install()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_install_preserves_bytes_and_existing_data(self):
        marker = self.root / 'labels.tsv'
        marker.write_text('synthetic test fixture, not patient data')
        self.install(apply=True)
        for canonical, entry in CATALOG.items():
            self.assertEqual(syndrodx.digest(self.root / canonical), entry['sha256'])
        self.assertEqual(marker.read_text(), 'synthetic test fixture, not patient data')
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.install(apply=True)
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_different_version_refused_before_any_writes(self):
        changed = self.root / 'build_labels.py'
        changed.write_text('sentinel = 1\n')
        with self.assertRaises(ValueError):
            self.install(apply=True)
        self.assertEqual(list(self.root.iterdir()), [changed])
        self.assertEqual(changed.read_text(), 'sentinel = 1\n')

    def test_explicit_replace_saves_previous_code(self):
        changed = self.root / 'build_labels.py'
        changed.write_text('sentinel = 1\n')
        self.install(apply=True, replace=True)
        backups = list(self.root.glob('syndrodx_code_backup_*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'build_labels.py').read_text(), 'sentinel = 1\n')
        self.assertEqual(syndrodx.digest(changed), CATALOG['build_labels.py']['sha256'])

    def test_symlink_destination_refused(self):
        with tempfile.TemporaryDirectory() as other:
            (self.root / 'face_qc').symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.install(apply=True)
            self.assertEqual(list(Path(other).iterdir()), [])

    def test_run_preview_does_not_execute_audit(self):
        self.install(apply=True)
        with contextlib.redirect_stdout(self.output):
            syndrodx.run(self.root, 'audit_metadata_sources', [], execute=False)
        self.assertFalse((self.root / 'metadata_audit').exists())

    def test_changed_dependency_blocks_execution(self):
        self.install(apply=True)
        (self.root / 'audit_metadata_sources.py').write_text('changed = True\n')
        with self.assertRaises(ValueError):
            syndrodx.run(self.root, 'plan_expert_splits', [], execute=False)


if __name__ == '__main__':
    unittest.main()
