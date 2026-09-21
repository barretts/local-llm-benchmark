"""Total inference deadlines cover continuous streams and detached sockets."""
import http.server
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from localbench.adapters.base import LocalHTTP
from localbench.schema import validate_tool


class HTTPDeadlineTests(unittest.TestCase):
    def test_continuous_chunks_cannot_reset_deadline_and_raw_is_preserved(self):
        clock = SimpleNamespace(now=0.0)
        chunk = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        def read(amount):
            clock.now += .1
            return chunk
        sock = SimpleNamespace(gettimeout=Mock(return_value=None), settimeout=Mock())
        response = SimpleNamespace(read1=Mock(side_effect=read), close=Mock(),
            fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock)))
        connection = SimpleNamespace(sock=None, close=Mock())
        client = LocalHTTP('http://127.0.0.1:12345', timeout=.25)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(client, 'request', return_value=(connection, response)), \
                patch('localbench.adapters.base.time.monotonic', side_effect=lambda: clock.now):
            prefix = Path(directory)/'deadline'
            result = client.measure({'stream':True}, prefix, validate_tool)
            self.assertFalse(result['valid_stream'])
            self.assertEqual(result['calls'], [])
            self.assertEqual(result['error'], 'TimeoutError:local_request_deadline_exceeded')
            self.assertEqual(response.read1.call_count, 3)
            self.assertEqual(Path(str(prefix)+'-raw.bin').read_bytes(), chunk*3)
            self.assertEqual(len(result.get('content', '')), 2)
            self.assertAlmostEqual(sock.settimeout.call_args_list[-1].args[0], .05)
        connection.close.assert_called_once()
        response.close.assert_called_once()

    def test_expired_header_request_closes_connection(self):
        sock = SimpleNamespace(gettimeout=Mock(return_value=10), settimeout=Mock())
        connection = SimpleNamespace(sock=sock, timeout=10,
            request=Mock(), getresponse=Mock(), close=Mock())
        with patch('localbench.adapters.base.http.client.HTTPConnection', return_value=connection), \
                patch('localbench.adapters.base.time.monotonic', side_effect=[0,0,2]):
            with self.assertRaisesRegex(TimeoutError, 'local_request_deadline_exceeded'):
                LocalHTTP('http://127.0.0.1:12345').request('/v1/chat/completions', {}, deadline=1)
        connection.close.assert_called_once()
        connection.getresponse.assert_not_called()

    def test_real_continuous_connection_close_stream_has_total_timeout(self):
        stop = threading.Event()
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                chunk = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                try:
                    while not stop.is_set():
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        stop.wait(.02)
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                client = LocalHTTP('http://127.0.0.1:'+str(server.server_port), timeout=.2)
                started = time.monotonic()
                result = client.measure({'stream':True}, Path(directory)/'real', validate_tool)
                self.assertLess(time.monotonic()-started, .8)
                self.assertFalse(result['valid_stream'])
                self.assertIn('TimeoutError:', result['error'])
                self.assertTrue(result['content'])
                self.assertEqual(result['calls'], [])
        finally:
            stop.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
