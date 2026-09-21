"""Run or resume only the explicitly authorized Qwen3.8 27B 16K experiment."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.focused_qwen38 import run

parser = argparse.ArgumentParser()
parser.add_argument('--serve', action='store_true')
args = parser.parse_args()
result = run(serve=args.serve)
print('16K qualification: ' + str(result.get('qualified_16k')), flush=True)
