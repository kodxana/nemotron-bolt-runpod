"""Exercise Runpod's unchanged handler against a local HTTP fixture, without a GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from scripts.build_checks import UPSTREAM_HANDLER_SHA256
ROOT = Path(__file__).resolve().parent.parent


class UpstreamHandlerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get('RUNPOD_SOURCE_DIR', '/src')) / 'handler.py'
        if not path.is_file():
            raise RuntimeError('Set RUNPOD_SOURCE_DIR to the original v2.30.0 src folder; see docs/MAINTAINER.md')
        if hashlib.sha256(path.read_bytes()).hexdigest() != UPSTREAM_HANDLER_SHA256:
            raise RuntimeError('Test source differs from the pinned Runpod handler')
        spec = importlib.util.spec_from_file_location('original_runpod_handler', path)
        cls.handler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.handler)

    async def asyncSetUp(self):
        self.requests = []
        self.status = 200
        self.reply = {'choices': [{'message': {'content': '4'}, 'finish_reason': 'stop'}],
                      'usage': {'completion_tokens': 1}}
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', self.respond)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        for name, value in {'VLLM_BASE_URL': str(server.make_url('')).rstrip('/'),
                            'REQUEST_TIMEOUT': 5, '_default_model_cache': None}.items():
            p = patch.object(self.handler, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.dict(os.environ, {'SERVED_MODEL_NAME': 'nemotron-bolt'})
        p.start()
        self.addCleanup(p.stop)

    async def respond(self, request):
        body = await request.json() if request.can_read_body else None
        self.requests.append((request.method, request.path, body))
        if isinstance(self.reply, str):
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            for offset in range(0, len(self.reply), 7):
                await response.write(self.reply[offset:offset + 7].encode())
            await response.write_eof()
            return response
        return web.json_response(self.reply, status=self.status)

    async def collect(self, data):
        return [item async for item in self.handler.handler({'id': 'fixture', 'input': data})]

    async def test_openai_chat_preserves_body_and_output(self):
        body = {'model': 'nemotron-bolt', 'messages': [{'role': 'user', 'content': 'Hi'}],
                'max_tokens': 32, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
                'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
                'response_format': {'type': 'json_object'}}
        result = await self.collect({'openai_route': '/v1/chat/completions', 'openai_input': body})
        self.assertEqual(self.requests, [('POST', '/v1/chat/completions', body)])
        self.assertEqual(result, [self.reply])

    async def test_models_route_is_get(self):
        self.reply = {'object': 'list', 'data': [{'id': 'nemotron-bolt', 'object': 'model'}]}
        self.assertEqual(await self.collect({'openai_route': '/v1/models'}), [self.reply])
        self.assertEqual(self.requests, [('GET', '/v1/models', None)])

    async def test_stream_remains_raw_sse_including_done(self):
        self.reply = 'data: {"choices":[{"delta":{"content":"4"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        body = {'model': 'nemotron-bolt', 'messages': [{'role': 'user', 'content': 'Hi'}],
                'stream': True, 'stream_options': {'include_usage': True}}
        parts = await self.collect({'openai_route': '/v1/chat/completions', 'openai_input': body})
        self.assertTrue(all(isinstance(part, str) for part in parts))
        self.assertEqual(''.join(parts), self.reply)
        self.assertEqual(self.requests[0][2], body)

    async def test_responses_and_messages_are_forwarded(self):
        for route, body in [('/v1/responses', {'model': 'nemotron-bolt', 'input': 'Hi'}),
                            ('/v1/messages', {'model': 'nemotron-bolt', 'max_tokens': 32,
                                              'messages': [{'role': 'user', 'content': 'Hi'}]})]:
            with self.subTest(route=route):
                await self.collect({'openai_route': route, 'openai_input': body})
                self.assertEqual(self.requests[-1], ('POST', route, body))

    async def test_native_messages_use_sampling_params_and_served_model(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        await self.collect({'messages': messages, 'sampling_params': {'max_tokens': 32, 'temperature': 0}})
        self.assertEqual(self.requests, [('POST', '/v1/chat/completions', {
            'model': 'nemotron-bolt', 'messages': messages, 'max_tokens': 32, 'temperature': 0, 'stream': False})])

    async def test_native_prompt_uses_completion_route(self):
        await self.collect({'prompt': 'Hi', 'sampling_params': {'max_tokens': 8}})
        self.assertEqual(self.requests, [('POST', '/v1/completions', {
            'model': 'nemotron-bolt', 'prompt': 'Hi', 'max_tokens': 8, 'stream': False})])

    async def test_sdk_retains_upstream_error_payload_then_recovers(self):
        from runpod.serverless.modules.rp_job import run_job_generator
        job = {'id': 'fixture', 'input': {'openai_route': '/v1/chat/completions',
               'openai_input': {'model': 'nemotron-bolt', 'messages': [], 'max_tokens': -1}}}
        self.status = 400
        self.reply = {'error': {'message': 'invalid tokens'}}
        result = [item async for item in run_job_generator(self.handler.handler, job)]
        self.assertIn('error', result[0]['output'])
        self.status = 200
        self.reply = {'choices': [{'message': {'content': 'ready'}}]}
        result = [item async for item in run_job_generator(self.handler.handler, job)]
        self.assertEqual(result[0]['output'], self.reply)

    async def test_hub_and_examples_use_stock_input_contract(self):
        inputs = [json.loads(path.read_text())['input'] for path in (ROOT / 'examples').glob('*.json')]
        inputs += [case['input'] for case in json.loads((ROOT / '.runpod/tests.json').read_text())['tests']]
        for data in inputs:
            with self.subTest(data=data):
                route, method, body = self.handler._normalize_job_input(data)
                if route == '/v1/models':
                    self.assertEqual((method, body), ('GET', None))
                else:
                    self.assertEqual((method, route), ('POST', '/v1/chat/completions'))
                    self.assertEqual(body['model'], 'nemotron-bolt')
                    self.assertGreater(body['max_tokens'], 0)
                    self.assertTrue(body['messages'])


if __name__ == '__main__':
    unittest.main()
