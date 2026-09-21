'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/store.js');
const CASES = {"trace":[{"event":{"id":"event-42-0","accountId":"constructor","sequence":1,"delta":-143},"expected":{"duplicate":false,"balance":-143}},{"event":{"id":"event-42-1","accountId":"alpha","sequence":1,"delta":179},"expected":{"duplicate":false,"balance":179}},{"event":{"id":"event-42-2","accountId":"beta","sequence":1,"delta":-75},"expected":{"duplicate":false,"balance":-75}},{"event":{"id":"event-42-3","accountId":"alpha","sequence":2,"delta":-129},"expected":{"duplicate":false,"balance":50}},{"event":{"id":"event-42-4","accountId":"constructor","sequence":2,"delta":-148},"expected":{"duplicate":false,"balance":-291}},{"event":{"id":"event-42-5","accountId":"constructor","sequence":3,"delta":179},"expected":{"duplicate":false,"balance":-112}},{"event":{"id":"event-42-6","accountId":"constructor","sequence":4,"delta":-156},"expected":{"duplicate":false,"balance":-268}},{"event":{"id":"event-42-7","accountId":"constructor","sequence":5,"delta":16},"expected":{"duplicate":false,"balance":-252}},{"event":{"id":"event-42-8","accountId":"alpha","sequence":3,"delta":-185},"expected":{"duplicate":false,"balance":-135}},{"event":{"id":"event-42-9","accountId":"alpha","sequence":4,"delta":-89},"expected":{"duplicate":false,"balance":-224}},{"event":{"id":"event-42-10","accountId":"alpha","sequence":5,"delta":58},"expected":{"duplicate":false,"balance":-166}},{"event":{"id":"event-42-11","accountId":"constructor","sequence":6,"delta":-187},"expected":{"duplicate":false,"balance":-439}},{"event":{"id":"event-42-12","accountId":"constructor","sequence":7,"delta":-99},"expected":{"duplicate":false,"balance":-538}},{"event":{"id":"event-42-13","accountId":"constructor","sequence":8,"delta":132},"expected":{"duplicate":false,"balance":-406}},{"event":{"id":"event-42-14","accountId":"constructor","sequence":9,"delta":79},"expected":{"duplicate":false,"balance":-327}},{"event":{"id":"event-42-15","accountId":"beta","sequence":2,"delta":-88},"expected":{"duplicate":false,"balance":-163}},{"event":{"id":"event-42-16","accountId":"beta","sequence":3,"delta":101},"expected":{"duplicate":false,"balance":-62}},{"event":{"id":"event-42-17","accountId":"beta","sequence":4,"delta":-197},"expected":{"duplicate":false,"balance":-259}},{"event":{"id":"event-42-18","accountId":"alpha","sequence":6,"delta":157},"expected":{"duplicate":false,"balance":-9}},{"event":{"id":"event-42-19","accountId":"beta","sequence":5,"delta":-26},"expected":{"duplicate":false,"balance":-285}},{"event":{"id":"event-42-20","accountId":"beta","sequence":6,"delta":-121},"expected":{"duplicate":false,"balance":-406}},{"event":{"id":"event-42-21","accountId":"alpha","sequence":7,"delta":190},"expected":{"duplicate":false,"balance":181}},{"event":{"id":"event-42-22","accountId":"beta","sequence":7,"delta":-148},"expected":{"duplicate":false,"balance":-554}},{"event":{"id":"event-42-23","accountId":"alpha","sequence":8,"delta":-6},"expected":{"duplicate":false,"balance":175}}],"balances":{"alpha":175,"beta":-554,"constructor":-327},"sequences":{"alpha":8,"beta":7,"constructor":9}};
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
