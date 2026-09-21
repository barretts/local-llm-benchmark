'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/page.js');
const CASES = {"records":[{"id":"r-42-23","createdAt":1},{"id":"r-42-25","createdAt":-1},{"id":"r-42-07","createdAt":5},{"id":"r-42-22","createdAt":4},{"id":"r-42-13","createdAt":0},{"id":"r-42-00","createdAt":-2},{"id":"r-42-19","createdAt":5},{"id":"r-42-15","createdAt":5},{"id":"r-42-05","createdAt":-1},{"id":"r-42-09","createdAt":3},{"id":"r-42-28","createdAt":1},{"id":"r-42-18","createdAt":0},{"id":"r-42-17","createdAt":5},{"id":"r-42-27","createdAt":2},{"id":"r-42-16","createdAt":-3},{"id":"r-42-21","createdAt":0},{"id":"r-42-14","createdAt":0},{"id":"r-42-01","createdAt":-3},{"id":"r-42-08","createdAt":-2},{"id":"r-42-20","createdAt":3},{"id":"r-42-11","createdAt":-3},{"id":"r-42-26","createdAt":3},{"id":"r-42-12","createdAt":-2},{"id":"r-42-02","createdAt":1},{"id":"r-42-03","createdAt":0},{"id":"r-42-10","createdAt":-3},{"id":"r-42-24","createdAt":-3},{"id":"r-42-06","createdAt":-2},{"id":"r-42-04","createdAt":0}],"ordered":[{"id":"r-42-01","createdAt":-3},{"id":"r-42-10","createdAt":-3},{"id":"r-42-11","createdAt":-3},{"id":"r-42-16","createdAt":-3},{"id":"r-42-24","createdAt":-3},{"id":"r-42-00","createdAt":-2},{"id":"r-42-06","createdAt":-2},{"id":"r-42-08","createdAt":-2},{"id":"r-42-12","createdAt":-2},{"id":"r-42-05","createdAt":-1},{"id":"r-42-25","createdAt":-1},{"id":"r-42-03","createdAt":0},{"id":"r-42-04","createdAt":0},{"id":"r-42-13","createdAt":0},{"id":"r-42-14","createdAt":0},{"id":"r-42-18","createdAt":0},{"id":"r-42-21","createdAt":0},{"id":"r-42-02","createdAt":1},{"id":"r-42-23","createdAt":1},{"id":"r-42-28","createdAt":1},{"id":"r-42-27","createdAt":2},{"id":"r-42-09","createdAt":3},{"id":"r-42-20","createdAt":3},{"id":"r-42-26","createdAt":3},{"id":"r-42-22","createdAt":4},{"id":"r-42-07","createdAt":5},{"id":"r-42-15","createdAt":5},{"id":"r-42-17","createdAt":5},{"id":"r-42-19","createdAt":5}],"limits":[1,2,4,7,1000]};
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
