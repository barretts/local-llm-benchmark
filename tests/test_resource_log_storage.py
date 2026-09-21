"""NTFS compression preserves raw records and works for subsequent appends."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from localbench.resources import compress_resource_log

@unittest.skipUnless(os.name=='nt','NTFS filesystem behavior requires Windows')
class ResourceLogStorageTests(unittest.TestCase):
    def test_compression_and_append_preserve_exact_unicode_null_and_evidence_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'owned-resource-log.jsonl'
            record=dict(kind='host',time=1234.5,host={'ram_used_bytes':123456,'cpu_percent':None},
                foreign_workload_overlap=True,raw='Ω: preserved GPU counter evidence')
            payload=(json.dumps(record,ensure_ascii=False)+'\n').encode('utf-8')*1000
            path.write_bytes(payload);before=hashlib.sha256(payload).hexdigest()
            self.assertTrue(compress_resource_log(path))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            self.assertTrue(path.stat().st_file_attributes&0x800)
            with path.open('ab') as stream:stream.write(payload)
            self.assertEqual(path.read_bytes(),payload*2)
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),2000);self.assertEqual(rows[-1],record)
    def test_fresh_empty_log_remains_compressed_after_streamed_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'owned-resource-log.jsonl';path.touch()
            self.assertTrue(compress_resource_log(path))
            with path.open('a',encoding='utf-8',buffering=1) as stream:
                for index in range(10):stream.write(json.dumps({'kind':'gpu','time':index,'memory.used':index})+'\n')
            self.assertTrue(path.stat().st_file_attributes&0x800)
            self.assertEqual([json.loads(x)['time'] for x in path.read_text().splitlines()],list(range(10)))

if __name__=='__main__':unittest.main()
