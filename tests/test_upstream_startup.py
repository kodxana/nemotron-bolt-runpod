"""Check our configuration through the actual pinned startup and argument builder."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import Mock, patch

from scripts.build_checks import UPSTREAM_SHA256

ROOT = Path(__file__).resolve().parent.parent


def docker_environment():
    # Only ENV instructions from this Dockerfile; no Docker daemon needed.
    text = (ROOT / 'Dockerfile').read_text(encoding='utf-8').replace('\\\n', ' ')
    env = {}
    for line in text.splitlines():
        if line.startswith('ENV '):
            for item in shlex.split(line[4:]):
                key, value = item.split('=', 1)
                env[key] = value
    return env


class UpstreamStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('RUNPOD_SOURCE_DIR', '/src'))
        for name, expected in UPSTREAM_SHA256.items():
            if not (source / name).is_file():
                raise RuntimeError('Set RUNPOD_SOURCE_DIR to v2.30.0 src; see docs/MAINTAINER.md')
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Unexpected upstream source: {name}')
        sys.path.insert(0, str(source))
        cls.addClassCleanup(sys.path.remove, str(source))
        cls.startup = importlib.import_module('main')
        cls.handler = importlib.import_module('handler')
        cls.builder = importlib.import_module('args_builder')
        cls.env = docker_environment()

    def test_original_entrypoint_and_model_metadata(self):
        docker = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
        self.assertIn('ENTRYPOINT ["/opt/queue/bin/python", "/src/main.py"]', docker)
        self.assertIn('COPY handler.py /src/handler.py', docker)
        self.assertIn('!handler.py', (ROOT / '.dockerignore').read_text(encoding='utf-8').splitlines())
        self.assertNotIn('COPY worker', docker)
        self.assertNotIn('PATH=', docker)
        for name in ('runtime.py', 'queue.py', 'config.py', 'assets.py'):
            self.assertFalse((ROOT / 'worker' / name).exists())
        with patch.dict(os.environ, {'MODEL_NAME': 'incorrect/remote', 'TOKENIZER_NAME': 'wrong'}, clear=True), \
             patch.object(self.startup, 'LOCAL_MODEL_ARGS_PATH', str(ROOT / 'local_model_args.json')):
            self.startup.apply_local_model_args()
            self.assertEqual(os.environ['MODEL_NAME'], '/runpod-volume/models/bolt')
            self.assertEqual(os.environ['TOKENIZER_NAME'], '/runpod-volume/models/bolt')
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')

    def test_baked_preflight_never_contacts_hub(self):
        with patch.object(self.startup.model_preflight, 'HfApi', side_effect=AssertionError('Unexpected network access')):
            self.assertIsNone(self.startup.model_preflight.check_model_access(self.env))

    def test_nemotron_recipe_reaches_vllm(self):
        argv = self.builder.build_vllm_args(self.env)
        for flag, value in {
            '--model': '/runpod-volume/models/bolt', '--tokenizer': '/runpod-volume/models/bolt',
            '--served-model-name': 'nemotron-bolt', '--max-model-len': '32768',
            '--max-num-seqs': '1', '--max-num-batched-tokens': '2048',
            '--gpu-memory-utilization': '0.90', '--tensor-parallel-size': '1', '--load-format': 'auto',
            '--linear-backend': 'b12x', '--moe-backend': 'b12x', '--kv-cache-dtype': 'fp8',
            '--attention-backend': 'FLASHINFER', '--mamba-backend': 'flashinfer',
            '--mamba-ssm-cache-dtype': 'bfloat16', '--mamba-cache-dtype': 'bfloat16', '--mamba-cache-mode': 'align',
            '--reasoning-parser-plugin': '/runpod-volume/models/bolt/ultra_v3_reasoning_parser.py',
            '--reasoning-parser': 'ultra_v3', '--tool-call-parser': 'qwen3_coder',
            '--generation-config': 'vllm', '--seed': '42', '--video-pruning-rate': '0.5',
        }.items():
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in ('--trust-remote-code', '--enable-chunked-prefill', '--enable-auto-tool-choice',
                     '--disable-uvicorn-access-log', '--enforce-eager'):
            self.assertIn(flag, argv)
        self.assertEqual(json.loads(argv[argv.index('--limit-mm-per-prompt') + 1]), {'video': 1, 'image': 1})
        self.assertEqual(json.loads(argv[argv.index('--media-io-kwargs') + 1]), {'video': {'fps': 2, 'num_frames': 256}})
        spec = json.loads(argv[argv.index('--speculative-config') + 1])
        self.assertEqual((spec['method'], spec['model'], spec['num_speculative_tokens']),
                         ('dflash', '/runpod-volume/models/lilicorr', 7))

    def test_hub_toggles_are_native_worker_settings(self):
        hub = json.loads((ROOT / '.runpod/hub.json').read_text(encoding='utf-8'))
        controls = {item['key']: item['input'] for item in hub['config']['env']}
        tests = json.loads((ROOT / '.runpod/tests.json').read_text(encoding='utf-8'))
        test_env = {item['key']: item['value'] for item in tests['config']['env']}
        context = controls['MAX_MODEL_LEN']
        self.assertEqual(hub['config']['gpuIds'],
                         'BLACKWELL_96,ADA_48_PRO,-NVIDIA L40,-NVIDIA L40S,-NVIDIA RTX 6000 Ada Generation')
        self.assertEqual(tests['config']['gpuTypeId'], 'NVIDIA RTX PRO 6000 Blackwell Server Edition')
        self.assertEqual(hub['config']['gpuCount'], 1)
        self.assertEqual(tests['config']['gpuCount'], 1)
        self.assertEqual(context['default'], 32768)
        self.assertLessEqual(context['min'], context['default'])
        for env in (self.env, {**self.env, **test_env}):
            argv = self.builder.build_vllm_args(env)
            self.assertEqual(int(argv[argv.index('--max-model-len') + 1]), context['default'])
            self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '1')
            self.assertIn('--enforce-eager', argv)
        self.assertNotIn('ENABLE_SPECULATIVE', controls)
        for eager in (True, False):
            for spec in (True, False):
                with self.subTest(eager=eager, speculative=spec):
                    env = {**self.env,
                           'ENFORCE_EAGER': controls['ENFORCE_EAGER']['trueValue' if eager else 'falseValue'],
                           'SPECULATIVE_CONFIG': controls['SPECULATIVE_CONFIG']['trueValue' if spec else 'falseValue']}
                    argv = self.builder.build_vllm_args(env)
                    self.assertEqual('--enforce-eager' in argv, eager)
                    self.assertEqual('--no-enforce-eager' in argv, not eager)
                    self.assertEqual('--speculative-config' in argv, spec)
        with self.assertRaises(ValueError):
            self.builder.build_vllm_args({**self.env, 'ENFORCE_EAGER': 'invalid'})

    def test_gpu_presets_translate_to_original_worker_arguments(self):
        config = json.loads((ROOT / '.runpod/hub.json').read_text(encoding='utf-8'))['config']
        controls = {item['key']: item['input'] for item in config['env']}
        expected_contexts = {
            'RTX PRO 6000 Blackwell 96GB - 32K (default)': 32768,
            'RTX PRO 6000 Blackwell MIG 48GB - 32K (retest required)': 32768,
            'RTX 5090 32GB - 2K (experimental)': 2048,
            'RTX PRO 4500 Blackwell 32GB - 2K (experimental)': 2048,
        }
        self.assertEqual({p['name'] for p in config['presets']}, set(expected_contexts))
        for key, value in config['presets'][0]['defaults'].items():
            self.assertEqual(value, controls[key]['default'])
        for preset in config['presets']:
            with self.subTest(preset=preset['name']):
                self.assertEqual(set(preset), {'name', 'defaults'})
                env = dict(self.env)
                for key, value in preset['defaults'].items():
                    control = controls[key]
                    if control['type'] == 'boolean':
                        self.assertIsInstance(value, bool)
                        env[key] = control['trueValue' if value else 'falseValue']
                    else:
                        self.assertGreaterEqual(value, control['min'])
                        self.assertLessEqual(value, control['max'])
                        env[key] = str(value)
                argv = self.builder.build_vllm_args(env)
                self.assertEqual(int(argv[argv.index('--max-model-len') + 1]), expected_contexts[preset['name']])
                self.assertIn('--enforce-eager', argv)
                spec = json.loads(argv[argv.index('--speculative-config') + 1])
                self.assertEqual(spec['model'], '/runpod-volume/models/lilicorr')
                self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '1')

    def test_gpu_pool_filters_only_admit_compatible_48gb_or_larger_cards(self):
        hub = json.loads((ROOT / '.runpod/hub.json').read_text(encoding='utf-8'))['config']
        tests = json.loads((ROOT / '.runpod/tests.json').read_text(encoding='utf-8'))['config']
        catalog = json.loads((ROOT / 'docs/gpu-catalog-snapshot.json').read_text(encoding='utf-8'))
        selectors = hub['gpuIds'].split(',')
        pools = {item for item in selectors if not item.startswith('-')}
        excluded = {item[1:] for item in selectors if item.startswith('-')}
        allowed = {gpu['id'] for gpu in catalog['gpus']
                   if gpu.get('pool') in pools and gpu['id'] not in excluded}
        expected = {
            'NVIDIA RTX PRO 6000 Blackwell Server Edition',
            'NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
            'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition',
            'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb',
        }
        self.assertEqual(allowed, expected)
        for gpu in catalog['gpus']:
            if gpu['id'] in allowed:
                self.assertGreaterEqual(gpu['memory'], 48)
        self.assertIn(tests['gpuTypeId'], allowed)
        self.assertIsInstance(tests['gpuTypeId'], str)
        self.assertNotIn('gpuTypeIds', tests)
        self.assertEqual(hub['allowedCudaVersions'], ['13.0', '13.1', '13.2', '13.3', '13.4'])
        self.assertEqual(tests['allowedCudaVersions'], hub['allowedCudaVersions'])

    def test_original_startup_waits_for_health_then_registers_original_handler(self):
        events = []
        proc = Mock()
        sdk = Mock()
        sdk.serverless.start.side_effect = lambda config: events.append('queue')
        with patch.dict(os.environ, self.env, clear=True), \
             patch.object(self.startup, 'LOCAL_MODEL_ARGS_PATH', str(ROOT / 'local_model_args.json')), \
             patch.object(self.startup.compile_cache, 'ensure_usable'), patch.object(self.startup.signal, 'signal'), \
             patch.object(self.startup, 'start_vllm', side_effect=lambda: events.append('engine') or proc), \
             patch.object(self.startup, 'wait_for_vllm', side_effect=lambda _: events.append('health')), \
             patch.dict(sys.modules, {'runpod': sdk}):
            self.startup.main()
        self.assertEqual(events, ['engine', 'health', 'queue'])
        config = sdk.serverless.start.call_args.args[0]
        self.assertIs(config['handler'], self.handler.handler)
        self.assertIs(self.handler.vllm_process, proc)
        self.assertEqual(config['concurrency_modifier'](30), 1)
        self.assertTrue(config['return_aggregate_stream'])

    def test_engine_launch_uses_global_cli_and_loopback(self):
        proc = Mock()
        with patch.dict(os.environ, self.env, clear=True), \
             patch.object(self.startup.subprocess, 'Popen', return_value=proc) as popen, \
             patch.object(self.startup.threading, 'Thread'):
            self.startup.start_vllm()
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:6], ['vllm', 'serve', '--host', '127.0.0.1', '--port', '8000'])
        self.assertIn('--enforce-eager', argv)
        self.assertEqual(popen.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
