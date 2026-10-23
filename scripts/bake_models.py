"""Bake and verify pinned model assets during the Docker build only."""
import hashlib
import json
import logging
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
LOCK = json.loads((ROOT / 'assets.lock.json').read_text(encoding='utf-8'))
MODEL_DIR = Path('/runpod-volume/models')

log = logging.getLogger(__name__)


def snapshot_path(root, spec):
    return root / spec['directory']


def validate_snapshot(path, spec):
    for item in spec['files']:
        target = path / item['path']
        if not target.is_file() or target.stat().st_size != item['size']:
            raise ValueError(f'Incomplete baked model: {spec["repo"]}/{item["path"]}; rebuild the image')
        if 'sha256' in item:
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != item['sha256']:
                raise ValueError(f'Model integrity failure: {item["path"]}; rebuild the image')
    return path



def bake(root, download=None, token=False):
    """Build-only entry point. local_dir creates ordinary files, not cache symlinks."""
    if download is None:
        from huggingface_hub import snapshot_download
        download = snapshot_download
    for name in ('model', 'draft'):
        spec = LOCK[name]
        destination = snapshot_path(root, spec)
        destination.mkdir(parents=True, exist_ok=True)
        log.info('Baking %s at revision %s', spec['repo'], spec['revision'])
        download(repo_id=spec['repo'], revision=spec['revision'],
                 local_dir=str(destination), allow_patterns=[f['path'] for f in spec['files']],
                 token=token)
        validate_snapshot(destination, spec)
        (destination / '.revision').write_text(spec['revision'] + '\n', encoding='utf-8')
        # Local download bookkeeping is not needed in the final image.
        shutil.rmtree(destination / '.cache', ignore_errors=True)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    secret = Path('/run/secrets/HF_TOKEN')
    bake(MODEL_DIR, token=secret.read_text(encoding='utf-8').strip() if secret.is_file() else False)
