"""Fail builds on engine drift or incompatible inherited dependency metadata."""
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import hashlib
import json

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

UPSTREAM_HANDLER_SHA256 = '69e14fbbe81ad4b27695f48da3089472faf940a69518e75913a5422b65abce00'
UPSTREAM_SHA256 = {
    'handler.py': UPSTREAM_HANDLER_SHA256,
    'main.py': '3b8b93461216f346f15e56d0633a350cb760f82312051e01ba2a091c575d644a',
    'args_builder.py': '6dcb1db7bc22b25d6ce5eb0da73dc3dd2efc047e689c868a1d6c020fd2f9a911',
    'download_model.py': '802b109e4178ecee12701cbfb580fbb07a735c7803dc9ddae4a48e98404dcf44',
    'model_preflight.py': '30e435438769914b24977b2e7692abbccd2515efc7aca674e0a2ff7441189ea9',
    'compile_cache.py': 'c2037f37632094ea2b6222823d7122f5f821ac7d0af8b77b62be37d122d7ab43',
    'startup_errors.py': '80a37d5ba07b0fee3f7ae909941433333290d092fe8dabe3c0dc572e390d098f',
}


def check_dependencies(name, extras=(), seen=None, errors=None):
    seen = set() if seen is None else seen
    top_level = errors is None
    errors = [] if errors is None else errors
    key = (canonicalize_name(name), tuple(sorted(extras)))
    if key in seen:
        return
    seen.add(key)
    try:
        requirements = distribution(name).requires or []
    except PackageNotFoundError:
        errors.append(f'Missing dependency: {name}')
        requirements = []
    for spec in requirements:
        req = Requirement(spec)
        if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in ('', *extras)):
            continue
        try:
            actual = version(req.name)
        except PackageNotFoundError:
            errors.append(f'{name} requires {req}; not installed')
            continue
        if actual not in req.specifier:
            errors.append(f'{name} requires {req}; installed {actual}')
        check_dependencies(req.name, req.extras, seen, errors)
    if top_level and errors:
        raise RuntimeError('Incompatible inference dependencies:\n' + '\n'.join(dict.fromkeys(errors)))


def main():
    for name, expected in UPSTREAM_SHA256.items():
        assert hashlib.sha256((Path('/src') / name).read_bytes()).hexdigest() == expected, f'Unexpected Runpod base source: {name}'
    assert version('b12x') == '1.5.1'
    assert version('torch').split('+')[0] == '2.13.0'
    lock = json.loads(Path('/app/assets.lock.json').read_text())
    assert lock['b12x'] == version('b12x')
    assert lock['vllm_version'] == version('vllm')
    check_dependencies('vllm')
    check_dependencies('b12x')
    print('Engine, kernel and recursive dependency metadata checks passed')


if __name__ == '__main__':
    main()
