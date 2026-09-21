'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/store.js');
const CASES = {"trace":[{"event":{"id":"event-314159-0","accountId":"alpha","sequence":1,"delta":-60},"expected":{"duplicate":false,"balance":-60}},{"event":{"id":"event-314159-1","accountId":"beta","sequence":1,"delta":-150},"expected":{"duplicate":false,"balance":-150}},{"event":{"id":"event-314159-2","accountId":"alpha","sequence":2,"delta":92},"expected":{"duplicate":false,"balance":32}},{"event":{"id":"event-314159-3","accountId":"beta","sequence":2,"delta":110},"expected":{"duplicate":false,"balance":-40}},{"event":{"id":"event-314159-4","accountId":"beta","sequence":3,"delta":61},"expected":{"duplicate":false,"balance":21}},{"event":{"id":"event-314159-5","accountId":"alpha","sequence":3,"delta":-114},"expected":{"duplicate":false,"balance":-82}},{"event":{"id":"event-314159-6","accountId":"constructor","sequence":1,"delta":114},"expected":{"duplicate":false,"balance":114}},{"event":{"id":"event-314159-7","accountId":"constructor","sequence":2,"delta":-183},"expected":{"duplicate":false,"balance":-69}},{"event":{"id":"event-314159-8","accountId":"alpha","sequence":4,"delta":57},"expected":{"duplicate":false,"balance":-25}},{"event":{"id":"event-314159-9","accountId":"constructor","sequence":3,"delta":102},"expected":{"duplicate":false,"balance":33}},{"event":{"id":"event-314159-10","accountId":"alpha","sequence":5,"delta":-105},"expected":{"duplicate":false,"balance":-130}},{"event":{"id":"event-314159-11","accountId":"beta","sequence":4,"delta":-199},"expected":{"duplicate":false,"balance":-178}},{"event":{"id":"event-314159-12","accountId":"constructor","sequence":4,"delta":-131},"expected":{"duplicate":false,"balance":-98}},{"event":{"id":"event-314159-13","accountId":"constructor","sequence":5,"delta":-67},"expected":{"duplicate":false,"balance":-165}},{"event":{"id":"event-314159-14","accountId":"beta","sequence":5,"delta":-48},"expected":{"duplicate":false,"balance":-226}},{"event":{"id":"event-314159-15","accountId":"beta","sequence":6,"delta":-31},"expected":{"duplicate":false,"balance":-257}},{"event":{"id":"event-314159-16","accountId":"constructor","sequence":6,"delta":81},"expected":{"duplicate":false,"balance":-84}},{"event":{"id":"event-314159-17","accountId":"beta","sequence":7,"delta":-95},"expected":{"duplicate":false,"balance":-352}},{"event":{"id":"event-314159-18","accountId":"alpha","sequence":6,"delta":-137},"expected":{"duplicate":false,"balance":-267}},{"event":{"id":"event-314159-19","accountId":"beta","sequence":8,"delta":-66},"expected":{"duplicate":false,"balance":-418}},{"event":{"id":"event-314159-20","accountId":"constructor","sequence":7,"delta":58},"expected":{"duplicate":false,"balance":-26}},{"event":{"id":"event-314159-21","accountId":"constructor","sequence":8,"delta":115},"expected":{"duplicate":false,"balance":89}},{"event":{"id":"event-314159-22","accountId":"alpha","sequence":7,"delta":177},"expected":{"duplicate":false,"balance":-90}},{"event":{"id":"event-314159-23","accountId":"alpha","sequence":8,"delta":-45},"expected":{"duplicate":false,"balance":-135}}],"balances":{"alpha":-135,"beta":-418,"constructor":89},"sequences":{"alpha":8,"beta":8,"constructor":8}};
const {EventStore} = subject;
const {Repository} = require('/workspace/.build/repository.js');
function fresh() { const repo = new Repository(); return {repo,store:new EventStore(repo)}; }
const event = (id,accountId,sequence,delta) => ({id,accountId,sequence,delta});
test('seeded cross-account event trace matches controller state after every event', () => {
  const {repo,store}=fresh();
  for (const row of CASES.trace) {
    const before=structuredClone(row.event); assert.deepEqual(store.apply(row.event),row.expected);
    assert.deepEqual(row.event,before); assert.equal(repo.has(row.event.id),true);
  }
  for (const [id,balance] of Object.entries(CASES.balances)) {
    assert.equal(store.balance(id),balance); assert.deepEqual(repo.read(id),{balance,lastSequence:CASES.sequences[id]});
  }
});
test('true duplicate returns current balance without reapplying', () => {
  const {store}=fresh(); const e=event('one','a',1,10); store.apply(e); store.apply(event('two','a',2,3));
  assert.deepEqual(store.apply(e),{duplicate:true,balance:13}); assert.equal(store.balance('a'),13);
});
test('globally duplicate changed payload precedes sequence validation', () => {
  const {store}=fresh(); store.apply(event('same','a',1,8)); store.apply(event('other','b',1,-3));
  assert.deepEqual(store.apply(event('same','b',NaN,Infinity)),{duplicate:true,balance:-3});
  assert.equal(store.balance('a'),8); assert.equal(store.balance('b'),-3);
});
test('each account has independent sequences and negative balances are valid', () => {
  const {store}=fresh(); assert.deepEqual(store.apply(event('a','x',1,-10)),{duplicate:false,balance:-10});
  assert.deepEqual(store.apply(event('b','y',1,2)),{duplicate:false,balance:2});
  assert.throws(()=>store.apply(event('c','x',3,1)),Error);
  assert.deepEqual(store.apply(event('c','x',2,-5)),{duplicate:false,balance:-15});
});
test('safe integer and sequence validation leave ids and accounts unmodified', () => {
  for (const [sequence,delta] of [[0,1],[-1,1],[1.5,1],[NaN,1],[Infinity,1],[1,0.5],[1,NaN],[1,Infinity],[1,Number.MAX_SAFE_INTEGER+1],[Number.MAX_SAFE_INTEGER+1,1]]) {
    const {repo,store}=fresh(); assert.throws(()=>store.apply(event('invalid','a',sequence,delta)),Error);
    assert.equal(repo.has('invalid'),false); assert.deepEqual(repo.read('a'),{balance:0,lastSequence:0});
  }
});
test('existing account commit failure rolls back state and seen together', () => {
  const {repo,store}=fresh(); store.apply(event('first','a',1,11)); repo.failNextCommit(); const e=event('next','a',2,-7);
  assert.throws(()=>store.apply(e),Error); assert.deepEqual(repo.read('a'),{balance:11,lastSequence:1}); assert.equal(repo.has('next'),false);
  assert.deepEqual(store.apply(e),{duplicate:false,balance:4});
});
test('repository read and transaction write never retain borrowed state', () => {
  const {repo,store}=fresh(); store.apply(event('first','a',1,5)); const state=repo.read('a'); state.balance=900; state.lastSequence=90;
  assert.deepEqual(repo.read('a'),{balance:5,lastSequence:1});
  const borrowed={balance:8,lastSequence:2}; repo.transaction(tx=>tx.write('b',borrowed)); borrowed.balance=99;
  assert.deepEqual(repo.read('b'),{balance:8,lastSequence:2});
  repo.transaction(tx=>{const r=tx.read('b');r.balance=444;}); assert.equal(repo.read('b').balance,8);
});
test('transaction callback failure rolls back every write and seen marker', () => {
  const {repo}=fresh(); const failure=new Error('callback');
  assert.throws(()=>repo.transaction(tx=>{tx.write('a',{balance:5,lastSequence:1});tx.markSeen('e');throw failure;}),error=>error===failure);
  assert.equal(repo.has('e'),false); assert.deepEqual(repo.read('a'),{balance:0,lastSequence:0});
});
test('failNextCommit affects one attempted commit and validation does not consume it', () => {
  const {repo,store}=fresh(); repo.failNextCommit(); assert.throws(()=>store.apply(event('bad','a',2,3)),Error);
  assert.throws(()=>store.apply(event('good','a',1,3)),Error); assert.equal(repo.has('good'),false);
  assert.deepEqual(store.apply(event('good','a',1,3)),{duplicate:false,balance:3});
  assert.deepEqual(store.apply(event('next','a',2,1)),{duplicate:false,balance:4});
});
test('frozen events stay unchanged and unknown account reads are independent', () => {
  const {repo,store}=fresh(); const e=Object.freeze(event('frozen','a',1,4)); const before=structuredClone(e);
  assert.deepEqual(store.apply(e),{duplicate:false,balance:4}); assert.deepEqual(e,before);
  const r=repo.read('missing'); r.balance=42; assert.deepEqual(repo.read('missing'),{balance:0,lastSequence:0});
});
