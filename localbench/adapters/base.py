import abc
import http.client
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from ..config import atomic_json
from ..metrics import Measurement
from ..streams import normalize_openai, normalize_ollama


class EngineAdapter(abc.ABC):
    @abc.abstractmethod
    def discover_capabilities(self): ...
    @abc.abstractmethod
    def launch(self, config): ...
    @abc.abstractmethod
    def health(self): ...
    @abc.abstractmethod
    def tokenize(self, messages, tools): ...
    @abc.abstractmethod
    def stream_chat(self, request, **kwargs): ...
    @abc.abstractmethod
    def reset_cache(self): ...
    @abc.abstractmethod
    def unload_owned(self): ...
    @abc.abstractmethod
    def metadata(self): ...


class LocalHTTP:
    def __init__(self, endpoint, timeout=900, credential_env=None):
        parts = urlsplit(endpoint)
        if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("inference endpoint must be credential-free loopback HTTP")
        self.port = parts.port
        self.timeout = timeout
        self.credential_env = credential_env

    @staticmethod
    def remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("local_request_deadline_exceeded")
        return remaining

    def clamp_timeout(self, conn, response, deadline):
        remaining = self.remaining(deadline)
        # HTTP/1.0 and Connection: close detach the response socket from conn.
        sock = conn.sock or getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            current = sock.gettimeout()
            sock.settimeout(min(remaining, self.timeout, current if current is not None else remaining))
        return remaining

    def request(self, path, body=None, *, deadline=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                         timeout=self.timeout if deadline is None else self.remaining(deadline))
        headers = {"Content-Type":"application/json"}
        if self.credential_env and os.environ.get(self.credential_env):
            headers["Authorization"] = "Bearer " + os.environ[self.credential_env]
        encoded = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        response = None
        try:
            if deadline is not None:
                conn.timeout = self.remaining(deadline)
            conn.request("GET" if body is None else "POST", path, body=encoded, headers=headers)
            if deadline is not None:
                self.clamp_timeout(conn, None, deadline)
            response = conn.getresponse()
            if deadline is not None:
                self.clamp_timeout(conn, response, deadline)
            if response.status != 200:
                # Do not read or echo unexpected content which could include secrets.
                raise RuntimeError(f"local_http_status_{response.status}")
            return conn, response
        except Exception:
            if response is not None:
                response.close()
            conn.close()
            raise

    def json(self, path, body=None):
        conn, response = self.request(path, body)
        try:
            return json.loads(response.read())
        finally:
            conn.close()

    def measure(self, request, log_prefix, validator, mode="fresh", protocol="openai"):
        log_prefix = Path(log_prefix)
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(str(log_prefix)+"-request.json", request)
        m = Measurement(mode=mode, tool_validator=validator)
        events, calls, content = [], [], []
        m.start()  # includes connection, send, queue and stream delivery.
        deadline = time.monotonic() + self.timeout
        conn = response = None
        try:
            conn, response = self.request("/api/chat" if protocol=="ollama" else "/v1/chat/completions",request,deadline=deadline)
            with Path(str(log_prefix)+"-raw.bin").open("wb") as raw, Path(str(log_prefix)+"-events.jsonl").open("w",encoding="utf-8") as stream:
                def chunks():
                    while True:
                        self.clamp_timeout(conn, response, deadline)
                        chunk = response.read1(4096)
                        if chunk:
                            raw.write(chunk)
                            raw.flush()
                        self.remaining(deadline)
                        if not chunk:
                            break
                        yield chunk
                normalize = normalize_ollama if protocol=="ollama" else normalize_openai
                for event in normalize(chunks(), validator=validator):
                    m.observe(event)
                    stamped = {"arrival_ns":time.perf_counter_ns(),"event":event}
                    stream.write(json.dumps(stamped,ensure_ascii=False,allow_nan=False)+"\n")
                    stream.flush()
                    events.append(event)
                    if event["type"] == "tool_call":
                        calls.append(event)
                    if event["type"] == "content":
                        content.append(event["text"])
            result = {"metrics":m.finish(),"calls":calls,"content":"".join(content),"events":events,"valid_stream":True}
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as e:
            result = {"metrics":m.finish(),"calls":[],"content":"".join(content),"valid_stream":False,"error":type(e).__name__+":"+str(e)}
        finally:
            if response is not None:
                response.close()
            if conn:
                conn.close()
        atomic_json(str(log_prefix)+"-result.json",result)
        return result
