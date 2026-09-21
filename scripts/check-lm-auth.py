"""Read-only local API check; never prints credentials or response bodies."""
from pathlib import Path
import json
import sys
import time
import urllib.error
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localbench.config import load, atomic_json


def main():
    status=None;result='unavailable';key=None
    try:
        key=Path(r'C:\Users\barrett\lmstudio-local.txt').read_text(encoding='utf-8-sig').strip()
        if not key or any(not 33<=ord(c)<=126 for c in key):
            result='credential_format_invalid'
        else:
            request=urllib.request.Request('http://127.0.0.1:1234/api/v1/models',headers={'Authorization':'Bearer '+key})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request,timeout=10) as response:status=response.status
            except urllib.error.HTTPError as error:
                status=error.code;error.close()
            result='accepted' if status==200 else 'rejected' if status in (401,403) else 'unavailable'
    except (OSError,ValueError,urllib.error.URLError):pass
    finally:key=None
    receipt={'engine':'lm-studio','auth_readiness':result,'http_status':status,'checked_at':time.time(),
             'probe':'read-only loopback /api/v1/models; response body not read or logged'}
    config=load();atomic_json(Path(config['paths']['artifacts'])/'lm-studio-auth-readiness.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
