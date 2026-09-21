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
test('timestamp ties span pages without losing ids', () => {
  const records = [{id:'c',createdAt:1},{id:'a',createdAt:1},{id:'b',createdAt:1}];
  assert.deepEqual(allPages(records, 1).map(x => x.id), ['a','b','c']);
});
test('input order remains unchanged', () => {
  const records = [{id:'b',createdAt:2},{id:'a',createdAt:1}];
  const before = structuredClone(records); paginate(records, {limit:1}); assert.deepEqual(records, before);
});
