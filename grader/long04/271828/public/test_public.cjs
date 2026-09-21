'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/request.js');
const CASES = {"delay":21,"url":"/tenant-271828/events","key":"  event-key-271828-3502  ","statuses":[429,503,429,200],"body":"response-98107"};
const {requestWithRetry} = subject;
const {TimeoutError} = require('/workspace/.build/errors.js');
function options(overrides={}) { const delays=[]; return {delays,value:{method:'GET',maxAttempts:3,baseDelayMs:5,sleep:async ms=>{delays.push(ms);},...overrides}}; }
test('unsafe POST is attempted once with no delay', async () => {
  let calls=0; const final={status:503,body:'busy'}; const o=options({method:'POST'});
  assert.equal(await requestWithRetry(async()=>{calls++;return final;},'/x',o.value),final);
  assert.equal(calls,1); assert.deepEqual(o.delays,[]);
});
test('ordinary errors propagate immediately with no final sleep', async () => {
  let calls=0;const failure=new Error('ordinary');const o=options();
  await assert.rejects(requestWithRetry(async()=>{calls++;throw failure;},'/x',o.value),error=>error===failure);
  assert.equal(calls,1);assert.deepEqual(o.delays,[]);
});
