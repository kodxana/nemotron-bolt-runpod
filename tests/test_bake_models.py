import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.bake_models import bake, snapshot_path, validate_snapshot



class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / 'models'
        content = b'{"model_type":"fixture"}'
        self.content = content
        self.spec = {'repo': 'test/model', 'revision': 'a' * 40, 'directory': 'bolt',
                     'files': [{'path': 'config.json', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}

    def populate(self, spec=None, content=None):
        path = snapshot_path(self.model_dir, spec or self.spec)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'config.json').write_bytes(content or self.content)
        (path / '.revision').write_text((spec or self.spec)['revision'])
        return path

    def test_partial_baked_snapshot_fails_closed(self):
        self.populate(content=b'partial')
        with self.assertRaisesRegex(ValueError, 'Incomplete baked model'):
            validate_snapshot(snapshot_path(self.model_dir, self.spec), self.spec)

    def test_metadata_hash_mismatch(self):
        path = self.populate(content=b'x' * len(self.content))
        with self.assertRaises(ValueError):
            validate_snapshot(path, self.spec)

    def test_missing_shard_is_not_complete(self):
        path = self.populate()
        self.spec['files'].append({'path': 'model.safetensors', 'size': 10})
        with self.assertRaises(ValueError):
            validate_snapshot(path, self.spec)

    def test_bake_pins_both_revisions_and_removes_download_metadata(self):
        draft = {**self.spec, 'repo': 'test/draft', 'revision': 'b' * 40, 'directory': 'lilicorr'}
        def download(**kwargs):
            destination = Path(kwargs['local_dir'])
            (destination / 'config.json').write_bytes(self.content)
            (destination / '.cache').mkdir()
            return str(destination)
        callback = Mock(side_effect=download)
        with patch('scripts.bake_models.LOCK', {'model': self.spec, 'draft': draft}):
            bake(self.model_dir, callback)
        self.assertEqual(callback.call_count, 2)
        for call, spec in zip(callback.call_args_list, (self.spec, draft)):
            self.assertEqual(call.kwargs['repo_id'], spec['repo'])
            self.assertEqual(call.kwargs['revision'], spec['revision'])
            self.assertEqual(call.kwargs['allow_patterns'], ['config.json'])
            self.assertFalse(call.kwargs['token'])
            self.assertFalse((Path(call.kwargs['local_dir']) / '.cache').exists())
        self.assertEqual((snapshot_path(self.model_dir, draft) / '.revision').read_text().strip(), draft['revision'])

    def test_bake_rejects_corrupt_weight_bytes(self):
        spec = {**self.spec, 'files': [{'path': 'weights.safetensors', 'size': 3,
                                      'sha256': hashlib.sha256(b'abc').hexdigest()}]}
        def download(**kwargs):
            (Path(kwargs['local_dir']) / 'weights.safetensors').write_bytes(b'bad')
        with patch('scripts.bake_models.LOCK', {'model': spec, 'draft': spec}), self.assertRaisesRegex(ValueError, 'integrity'):
            bake(self.model_dir, download)

    def test_build_download_failure_is_not_ignored(self):
        callback = Mock(side_effect=OSError('interrupted'))
        with patch('scripts.bake_models.LOCK', {'model': self.spec, 'draft': self.spec}), self.assertRaises(OSError):
            bake(self.model_dir, callback)
        self.assertEqual(callback.call_count, 1)


if __name__ == '__main__':
    unittest.main()
