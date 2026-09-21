'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/cache.js');
const CASES = {"ttl":53,"keys":["key-271828-0","key-271828-1","key-271828-2","key-271828-3"],"values":[654,994,-688,861]};
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
async function tick() { for (let i = 0; i < 6; i++) await Promise.resolve(); }
const {AsyncTTLCache} = subject;
test('concurrent calls coalesce until resolution and values are reused', async () => {
  const d = deferred(); let calls = 0;
  const cache = new AsyncTTLCache(() => { calls++; return d.promise; }, CASES.ttl, () => 0);
  const waiters = Array.from({length: 12}, () => cache.get(CASES.keys[0]));
  await tick(); assert.equal(calls, 1); d.resolve(CASES.values[0]);
  assert.deepEqual(await Promise.all(waiters), Array(12).fill(CASES.values[0]));
  assert.equal(await cache.get(CASES.keys[0]), CASES.values[0]); assert.equal(calls, 1);
});
test('exact resolution deadline expires', async () => {
  let now = 0, calls = 0;
  const cache = new AsyncTTLCache(async () => ++calls, CASES.ttl, () => now);
  assert.equal(await cache.get('a'), 1); now = CASES.ttl - 1;
  assert.equal(await cache.get('a'), 1); now = CASES.ttl;
  assert.equal(await cache.get('a'), 2);
});
test('failed pending load is removed, including coalesced rejection', async () => {
  let calls = 0; const d = deferred(), failure = new Error('seeded failure');
  const cache = new AsyncTTLCache(() => ++calls === 1 ? d.promise : Promise.resolve(CASES.values[1]), CASES.ttl, () => 0);
  const a = cache.get('a'), b = cache.get('a');
  const errors = Promise.allSettled([a, b]); await tick(); d.reject(failure);
  assert.deepEqual((await errors).map(x => x.reason), [failure, failure]);
  assert.equal(await cache.get('a'), CASES.values[1]); assert.equal(calls, 2);
});
test('keys have independent pending loads and expiries', async () => {
  let now = 0; const calls = new Map();
  const cache = new AsyncTTLCache(async key => { calls.set(key, (calls.get(key) || 0) + 1); return key; }, CASES.ttl, () => now);
  assert.deepEqual(await Promise.all(CASES.keys.map(key => cache.get(key))), CASES.keys);
  cache.invalidate(CASES.keys[1]); assert.equal(await cache.get(CASES.keys[1]), CASES.keys[1]);
  assert.equal(await cache.get(CASES.keys[0]), CASES.keys[0]);
  assert.equal(calls.get(CASES.keys[0]), 1); assert.equal(calls.get(CASES.keys[1]), 2);
});
test('invalidated stale loads cannot overwrite newer out-of-order resolutions', async () => {
  const old = deferred(), fresh = deferred(); let calls = 0;
  const cache = new AsyncTTLCache(() => ++calls === 1 ? old.promise : fresh.promise, CASES.ttl, () => 0);
  const oldWaiter = cache.get('x'); await tick(); cache.invalidate('x');
  const newWaiter = cache.get('x'); await tick(); fresh.resolve(22);
  assert.equal(await newWaiter, 22); old.resolve(11); assert.equal(await oldWaiter, 11);
  assert.equal(await cache.get('x'), 22); assert.equal(calls, 2);
});
test('repeated invalidation excludes stale values without rejecting waiters', async () => {
  const pending = [deferred(), deferred(), deferred()]; let calls = 0;
  const cache = new AsyncTTLCache(() => pending[calls++].promise, CASES.ttl, () => 0);
  const a = cache.get('x'); await tick(); cache.invalidate('x'); cache.invalidate('x');
  const b = cache.get('x'); await tick(); cache.invalidate('x');
  pending[0].resolve(1); pending[1].resolve(2); assert.deepEqual(await Promise.all([a, b]), [1, 2]);
  const c = cache.get('x'); await tick(); pending[2].resolve(3);
  assert.equal(await c, 3); assert.equal(calls, 3);
});
test('TTL rejects nonpositive and nonfinite values', () => {
  for (const ttl of [0, -1, NaN, Infinity, -Infinity])
    assert.throws(() => new AsyncTTLCache(async () => 1, ttl, () => 0), Error);
});
test('seeded slow resolutions retain full TTL on all keys', async () => {
  for (let i = 0; i < CASES.keys.length; i++) {
    let now = 0, calls = 0; const d = deferred();
    const cache = new AsyncTTLCache(() => ++calls === 1 ? d.promise : Promise.resolve(-99999), CASES.ttl, () => now);
    const p = cache.get(CASES.keys[i]); await tick(); now = CASES.ttl * 3;
    d.resolve(CASES.values[i]); await p; now += CASES.ttl - 1;
    assert.equal(await cache.get(CASES.keys[i]), CASES.values[i]); assert.equal(calls, 1);
  }
});
