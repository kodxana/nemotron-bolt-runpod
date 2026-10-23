import unittest

from scripts.acceptance import stream_events


class AcceptanceParserTests(unittest.TestCase):
    def test_original_handler_chunk_boundaries_and_completion_marker(self):
        wire = ': keepalive\r\n\r\ndata: {"choices":[]}\r\n\r\ndata: [DONE]\r\n\r\n'
        self.assertEqual(stream_events([wire[i:i + 3] for i in range(0, len(wire), 3)]), [{'choices': []}])
        with self.assertRaisesRegex(AssertionError, 'completion marker'):
            stream_events(['data: {"choices":[]}\n\n'])
