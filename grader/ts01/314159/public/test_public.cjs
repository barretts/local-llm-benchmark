'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/cache.js');
const CASES = {"ttl":22,"keys":["key-314159-0","key-314159-1","key-314159-2","key-314159-3"],"values":[-438,-413,-798,-759]};
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
async function tick() { for (let i = 0; i < 6; i++) await Promise.resolve(); }
const {AsyncTTLCache} = subject;
test('a rejected load is retried', async () => {
  let calls = 0;
  const failure = new Error('loader failure');
  const cache = new AsyncTTLCache(async () => { if (++calls === 1) throw failure; return 17; }, 10, () => 0);
  await assert.rejects(cache.get('a'), error => error === failure);
  assert.equal(await cache.get('a'), 17);
  assert.equal(calls, 2);
});
test('a slow load receives its full resolution-time TTL', async () => {
  let now = 0, calls = 0;
  const d = deferred();
  const cache = new AsyncTTLCache(() => { calls++; return calls === 1 ? d.promise : Promise.resolve(99); }, 10, () => now);
  const first = cache.get('a'); await tick(); now = 20; d.resolve(7);
  assert.equal(await first, 7); now = 29;
  assert.equal(await cache.get('a'), 7);
  assert.equal(calls, 1);
});
