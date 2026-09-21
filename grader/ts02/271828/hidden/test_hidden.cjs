'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/page.js');
const CASES = {"records":[{"id":"r-271828-28","createdAt":-2},{"id":"r-271828-14","createdAt":2},{"id":"r-271828-00","createdAt":-1},{"id":"r-271828-21","createdAt":2},{"id":"r-271828-15","createdAt":1},{"id":"r-271828-04","createdAt":-1},{"id":"r-271828-25","createdAt":-3},{"id":"r-271828-24","createdAt":-1},{"id":"r-271828-05","createdAt":0},{"id":"r-271828-23","createdAt":0},{"id":"r-271828-12","createdAt":3},{"id":"r-271828-22","createdAt":1},{"id":"r-271828-13","createdAt":-1},{"id":"r-271828-20","createdAt":2},{"id":"r-271828-03","createdAt":-3},{"id":"r-271828-10","createdAt":0},{"id":"r-271828-17","createdAt":4},{"id":"r-271828-01","createdAt":-2},{"id":"r-271828-06","createdAt":-3},{"id":"r-271828-19","createdAt":4},{"id":"r-271828-11","createdAt":1},{"id":"r-271828-08","createdAt":-3},{"id":"r-271828-26","createdAt":3},{"id":"r-271828-16","createdAt":-2},{"id":"r-271828-07","createdAt":-1},{"id":"r-271828-02","createdAt":3},{"id":"r-271828-27","createdAt":5},{"id":"r-271828-18","createdAt":-1},{"id":"r-271828-09","createdAt":5}],"ordered":[{"id":"r-271828-03","createdAt":-3},{"id":"r-271828-06","createdAt":-3},{"id":"r-271828-08","createdAt":-3},{"id":"r-271828-25","createdAt":-3},{"id":"r-271828-01","createdAt":-2},{"id":"r-271828-16","createdAt":-2},{"id":"r-271828-28","createdAt":-2},{"id":"r-271828-00","createdAt":-1},{"id":"r-271828-04","createdAt":-1},{"id":"r-271828-07","createdAt":-1},{"id":"r-271828-13","createdAt":-1},{"id":"r-271828-18","createdAt":-1},{"id":"r-271828-24","createdAt":-1},{"id":"r-271828-05","createdAt":0},{"id":"r-271828-10","createdAt":0},{"id":"r-271828-23","createdAt":0},{"id":"r-271828-11","createdAt":1},{"id":"r-271828-15","createdAt":1},{"id":"r-271828-22","createdAt":1},{"id":"r-271828-14","createdAt":2},{"id":"r-271828-20","createdAt":2},{"id":"r-271828-21","createdAt":2},{"id":"r-271828-02","createdAt":3},{"id":"r-271828-12","createdAt":3},{"id":"r-271828-26","createdAt":3},{"id":"r-271828-17","createdAt":4},{"id":"r-271828-19","createdAt":4},{"id":"r-271828-09","createdAt":5},{"id":"r-271828-27","createdAt":5}],"limits":[1,2,4,7,1000]};
const {paginate} = subject;
const encode = (time, id) => Buffer.from(JSON.stringify([time, id]), 'utf8').toString('base64url');
function allPages(records, limit) {
  const found = []; let cursor;
  for (let i = 0; i <= records.length + 1; i++) {
    const page = paginate(records, {limit, ...(cursor === undefined ? {} : {cursor})});
    assert.ok(Array.isArray(page.items)); found.push(...page.items);
    if (page.nextCursor === null) return found;
    assert.equal(typeof page.nextCursor, 'string'); cursor = page.nextCursor;
  }
  assert.fail('pagination did not terminate');
}
test('seeded complete pagination equals the tuple oracle at all limits', () => {
  for (const limit of CASES.limits) assert.deepEqual(allPages(structuredClone(CASES.records), limit), CASES.ordered);
});
test('a deleted cursor record still filters by its tuple', () => {
  const data = [{id:'a',createdAt:2},{id:'c',createdAt:2},{id:'a',createdAt:3}];
  assert.deepEqual(paginate(data, {limit:10,cursor:encode(2,'b')}).items, [data[1],data[2]]);
});
test('Unicode ids use deterministic code-unit ordering and UTF-8 cursors', () => {
  const ids = ['é','Z','a','😀','ä','A','中','𝄞'];
  const data = ids.map(id => ({id, createdAt:9}));
  const expected = [...ids].sort((a,b)=>a<b?-1:a>b?1:0);
  assert.deepEqual(allPages(data, 1).map(x=>x.id), expected);
  const p = paginate(data, {limit:3});
  assert.equal(p.nextCursor, encode(9,expected[2]));
});
test('malformed cursor encodings and invalid tuple types throw', () => {
  const wrong = ['', '!', 'a=', 'a', '____', encode(1, 'a') + '='];
  for (const value of [null, {}, [1], [1,'a',3], ['1','a'], [1,3], [null,'a']])
    wrong.push(Buffer.from(JSON.stringify(value)).toString('base64url'));
  for (const cursor of wrong) assert.throws(()=>paginate([], {limit:1,cursor}), Error);
});
test('limit endpoints work and invalid limits fail', () => {
  assert.equal(paginate(CASES.records, {limit:1}).items.length, 1);
  assert.equal(paginate(CASES.records, {limit:1000}).items.length, CASES.records.length);
  for (const limit of [0,-1,1001,1.5,NaN,Infinity]) assert.throws(()=>paginate([], {limit}), Error);
});
test('exhausted and empty pages have null cursors', () => {
  assert.deepEqual(paginate([], {limit:2}), {items:[],nextCursor:null});
  assert.deepEqual(paginate(CASES.records, {limit:2,cursor:encode(99999,'last')}), {items:[],nextCursor:null});
  const last = CASES.ordered[CASES.ordered.length-1];
  assert.deepEqual(paginate(CASES.records, {limit:1,cursor:encode(last.createdAt,last.id)}), {items:[],nextCursor:null});
});
test('frozen unsorted source records are never changed', () => {
  const records = CASES.records.map(x=>Object.freeze({...x})); Object.freeze(records);
  const before = structuredClone(records); paginate(records, {limit:5}); assert.deepEqual(records,before);
});
test('distinct timestamps use exact last-item cursor and no extra cursor', () => {
  const data = [{id:'x',createdAt:7},{id:'z',createdAt:-1},{id:'y',createdAt:3}];
  const p = paginate(data,{limit:2}); assert.deepEqual(p.items,[data[1],data[2]]);
  assert.equal(p.nextCursor,encode(3,'y'));
  assert.deepEqual(paginate(data,{limit:2,cursor:p.nextCursor}),{items:[data[0]],nextCursor:null});
});
