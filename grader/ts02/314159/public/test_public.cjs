'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/page.js');
const CASES = {"records":[{"id":"r-314159-06","createdAt":1},{"id":"r-314159-13","createdAt":5},{"id":"r-314159-21","createdAt":2},{"id":"r-314159-25","createdAt":5},{"id":"r-314159-11","createdAt":0},{"id":"r-314159-22","createdAt":1},{"id":"r-314159-05","createdAt":1},{"id":"r-314159-12","createdAt":5},{"id":"r-314159-01","createdAt":1},{"id":"r-314159-02","createdAt":1},{"id":"r-314159-18","createdAt":5},{"id":"r-314159-15","createdAt":-1},{"id":"r-314159-10","createdAt":-3},{"id":"r-314159-14","createdAt":-1},{"id":"r-314159-04","createdAt":-2},{"id":"r-314159-23","createdAt":4},{"id":"r-314159-27","createdAt":0},{"id":"r-314159-00","createdAt":0},{"id":"r-314159-26","createdAt":2},{"id":"r-314159-20","createdAt":1},{"id":"r-314159-07","createdAt":5},{"id":"r-314159-19","createdAt":-1},{"id":"r-314159-24","createdAt":2},{"id":"r-314159-16","createdAt":2},{"id":"r-314159-17","createdAt":-3},{"id":"r-314159-08","createdAt":-1},{"id":"r-314159-09","createdAt":-1},{"id":"r-314159-03","createdAt":-2},{"id":"r-314159-28","createdAt":-2}],"ordered":[{"id":"r-314159-10","createdAt":-3},{"id":"r-314159-17","createdAt":-3},{"id":"r-314159-03","createdAt":-2},{"id":"r-314159-04","createdAt":-2},{"id":"r-314159-28","createdAt":-2},{"id":"r-314159-08","createdAt":-1},{"id":"r-314159-09","createdAt":-1},{"id":"r-314159-14","createdAt":-1},{"id":"r-314159-15","createdAt":-1},{"id":"r-314159-19","createdAt":-1},{"id":"r-314159-00","createdAt":0},{"id":"r-314159-11","createdAt":0},{"id":"r-314159-27","createdAt":0},{"id":"r-314159-01","createdAt":1},{"id":"r-314159-02","createdAt":1},{"id":"r-314159-05","createdAt":1},{"id":"r-314159-06","createdAt":1},{"id":"r-314159-20","createdAt":1},{"id":"r-314159-22","createdAt":1},{"id":"r-314159-16","createdAt":2},{"id":"r-314159-21","createdAt":2},{"id":"r-314159-24","createdAt":2},{"id":"r-314159-26","createdAt":2},{"id":"r-314159-23","createdAt":4},{"id":"r-314159-07","createdAt":5},{"id":"r-314159-12","createdAt":5},{"id":"r-314159-13","createdAt":5},{"id":"r-314159-18","createdAt":5},{"id":"r-314159-25","createdAt":5}],"limits":[1,2,4,7,1000]};
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
test('timestamp ties span pages without losing ids', () => {
  const records = [{id:'c',createdAt:1},{id:'a',createdAt:1},{id:'b',createdAt:1}];
  assert.deepEqual(allPages(records, 1).map(x => x.id), ['a','b','c']);
});
test('input order remains unchanged', () => {
  const records = [{id:'b',createdAt:2},{id:'a',createdAt:1}];
  const before = structuredClone(records); paginate(records, {limit:1}); assert.deepEqual(records, before);
});
