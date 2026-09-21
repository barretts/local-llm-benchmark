#!/bin/sh
set -eu
cd /workspace
tsc --strict --target ES2022 --module commonjs --outDir /workspace/.build /workspace/src/*.ts
node --test --test-reporter=/driver/node_reporter.cjs /tests/test_*.cjs
