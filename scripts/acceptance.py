"""Real Runpod acceptance; invokes an existing endpoint and may incur GPU charges."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
import urllib.request


def api(endpoint, operation, payload=None):
    url = f'https://api.runpod.ai/v2/{endpoint}/{operation}'
    headers = {'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']}
    data = None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def run(endpoint, payload, seconds):
    started = time.monotonic()
    job = api(endpoint, 'run', {'input': payload})
    job_id = job['id']
    try:
        while time.monotonic() - started < seconds:
            job = api(endpoint, f'status/{job_id}')
            if job['status'] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
                job['observed_wall_seconds'] = round(time.monotonic() - started, 3)
                return job
            time.sleep(2)
        raise TimeoutError('Acceptance deadline exceeded')
    except BaseException:
        # Cancel only the job created by this invocation; do not stop workers/endpoints.
        try:
            api(endpoint, f'cancel/{job_id}', {})
        except Exception:
            print(f'Could not cancel acceptance job {job_id}; inspect it in Runpod.')
        raise


def completion(job):
    assert job['status'] == 'COMPLETED', f'Job {job.get("id")} status {job["status"]}'
    output = job['output']
    assert isinstance(output, list) and len(output) == 1, 'Expected one completion in aggregate output'
    assert output[0]['choices'][0]['finish_reason'] != 'length', 'Output was truncated'
    return output[0]


def stream_events(chunks):
    assert all(isinstance(chunk, str) for chunk in chunks), 'Expected original handler SSE strings'
    events = []
    done = False
    for line in ''.join(chunks).splitlines():
        if line.startswith('data:'):
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            events.append(json.loads(data))
    assert done, 'SSE stream has no completion marker'
    return events


def chat_job(body):
    return {'openai_route': '/v1/chat/completions', 'openai_input': body}


def openai_stream(endpoint, body):
    request = urllib.request.Request(
        f'https://api.runpod.ai/v2/{endpoint}/openai/v1/chat/completions',
        data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'], 'Content-Type': 'application/json',
        })
    with urllib.request.urlopen(request, timeout=600) as response:
        return stream_events([response.read().decode('utf-8')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', type=Path, default=Path('validation-runpod.json'))
    parser.add_argument('--timeout', type=int, default=2100)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.endpoint):
        parser.error('Invalid endpoint ID')
    records = []

    def invoke(name, payload):
        job = run(args.endpoint, payload, args.timeout)
        records.append({'name': name, 'job': job})
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(name, job['status'], job['id'])
        return job

    text = {'model': 'nemotron-bolt',
            'messages': [{'role': 'user', 'content': 'What is 2 plus 2? Reply with one digit.'}],
            'temperature': 0, 'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}}
    for n in range(3):
        result = completion(invoke(f'text-{n+1}', chat_job(text)))
        assert result['choices'][0]['message']['content'].strip() == '4'
        assert result['usage']['completion_tokens'] > 0
    invalid = invoke('invalid-input', chat_job({**text, 'max_tokens': -1}))
    assert invalid['status'] == 'FAILED' or (
        invalid['status'] == 'COMPLETED' and any('error' in item for item in invalid.get('output', [])))
    assert completion(invoke('recovery', chat_job(text)))['choices'][0]['message']['content'].strip() == '4'
    stream_body = {**text, 'stream': True, 'stream_options': {'include_usage': True}}
    streamed = invoke('stream', chat_job(stream_body))
    assert streamed['status'] == 'COMPLETED'
    chunks = stream_events(streamed['output'])
    content = ''.join(c.get('delta', {}).get('content') or '' for event in chunks for c in event['choices'])
    assert content.strip() == '4'
    assert any(event.get('usage', {}).get('completion_tokens', 0) > 0 for event in chunks if event.get('usage'))
    tool = json.loads((Path(__file__).resolve().parent.parent / 'examples/tool-call.json').read_text())['input']
    call = completion(invoke('tool-call', tool))['choices'][0]['message']['tool_calls'][0]
    assert call['function']['name'] == 'get_temperature'
    assert 'warsaw' in json.loads(call['function']['arguments'])['city'].lower()
    # Small red PNG, generated locally with the standard library.
    import struct
    import zlib
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 64) * 64)) + chunk(b'IEND', b'')
    vision = {'model': 'nemotron-bolt', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Name the single dominant color in this image. One word.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
    ]}], 'max_tokens': 32, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    assert 'red' in completion(invoke('vision', chat_job(vision)))['choices'][0]['message']['content'].lower()

    models = api(args.endpoint, 'openai/v1/models')
    records.append({'name': 'openai-models', 'result': models})
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    assert any(model['id'] == 'nemotron-bolt' for model in models['data'])
    result = api(args.endpoint, 'openai/v1/chat/completions', text)
    records.append({'name': 'openai-chat', 'result': result})
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    assert result['choices'][0]['message']['content'].strip() == '4'
    events = openai_stream(args.endpoint, stream_body)
    records.append({'name': 'openai-stream', 'events': events})
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    content = ''.join(c.get('delta', {}).get('content') or '' for event in events for c in event.get('choices', []))
    assert content.strip() == '4'
    print('Acceptance passed. Keep this evidence with the exact image digest and GPU details.')


if __name__ == '__main__':
    main()
