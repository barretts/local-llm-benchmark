"""Read PE import names without executing downloaded code."""
from pathlib import Path
import json
import struct
import sys

def imports(path):
    data=Path(path).read_bytes()
    pe=struct.unpack_from('<I',data,0x3c)[0]
    if data[pe:pe+4]!=b'PE\0\0':raise ValueError('not a PE file')
    count=struct.unpack_from('<H',data,pe+6)[0]
    optional_size=struct.unpack_from('<H',data,pe+20)[0]
    optional=pe+24
    magic=struct.unpack_from('<H',data,optional)[0]
    directory=optional+(112 if magic==0x20b else 96)
    import_rva=struct.unpack_from('<I',data,directory+8)[0]
    sections=[]
    for i in range(count):
        entry=optional+optional_size+i*40
        vsize,rva,size,raw=struct.unpack_from('<IIII',data,entry+8)
        sections.append((rva,max(vsize,size),raw))
    def offset(rva):
        for base,size,raw in sections:
            if base<=rva<base+size:return raw+rva-base
        raise ValueError('invalid RVA')
    result=[]
    if import_rva:
        entry=offset(import_rva)
        while True:
            values=struct.unpack_from('<IIIII',data,entry)
            if not any(values):break
            start=offset(values[3]);end=data.index(0,start)
            result.append(data[start:end].decode('ascii'))
            entry+=20
    return result

print(json.dumps({str(Path(p).resolve()):imports(p) for p in sys.argv[1:]},indent=2))
