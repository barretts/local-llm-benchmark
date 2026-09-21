'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/request.js');
const CASES = {"delay":20,"url":"/tenant-42/events","key":"  event-key-42-2824  ","statuses":[429,503,429,200],"body":"response-39256"};
const {requestWithRetry} = subject;
const {TimeoutError} = require('/workspace/.build/errors.js');
function options(overrides={}) { const delays=[]; return {delays,value:{method:'GET',maxAttempts:3,baseDelayMs:5,sleep:async ms=>{delays.push(ms);},...overrides}}; }
test('GET retries only 429 and 503 with exact exponential pre-retry delays', async () => {
  const o=options({maxAttempts:4,baseDelayMs:CASES.delay});let calls=0;
  const responses=CASES.statuses.map(status=>({status,body:CASES.body}));
  assert.equal(await requestWithRetry(async()=>responses[calls++],CASES.url,o.value),responses[3]);
  assert.equal(calls,4);assert.deepEqual(o.delays,[CASES.delay,CASES.delay*2,CASES.delay*4]);
});
test('POST nonempty key is preserved on every fresh request and header object', async () => {
  const o=options({method:'POST',idempotencyKey:CASES.key});const requests=[],headers=[];
  const final={status:200,body:'ok'};
  const result=await requestWithRetry(async request=>{
    requests.push(request);headers.push(request.headers);
    assert.equal(request.url,CASES.url);assert.equal(request.method,'POST');
    assert.deepEqual(request.headers,{'Idempotency-Key':CASES.key});
    request.headers.changed='in transport';
    return requests.length<3?{status:503,body:'busy'}:final;
  },CASES.url,o.value);
  assert.equal(result,final);assert.equal(new Set(requests).size,3);assert.equal(new Set(headers).size,3);
  assert.deepEqual(o.delays,[5,10]);assert.equal(o.value.idempotencyKey,CASES.key);
});
test('empty missing and whitespace-only POST keys disable every retry', async () => {
  for (const key of [undefined,'',' \t\n ']) {
    let calls=0;const o=options({method:'POST',idempotencyKey:key});const failure=new TimeoutError('unsafe');
    await assert.rejects(requestWithRetry(async request=>{calls++;assert.deepEqual(request.headers,{});throw failure;},'/x',o.value),error=>error===failure);
    assert.equal(calls,1);assert.deepEqual(o.delays,[]);
  }
});
test('final retryable response returns unchanged and never sleeps after last attempt', async () => {
  for (const status of [429,503]) for (const maxAttempts of [1,3]) {
    let calls=0;const final={status,body:'same object'};const o=options({maxAttempts});
    assert.equal(await requestWithRetry(async()=>{calls++;return final;},'/x',o.value),final);
    assert.equal(calls,maxAttempts);assert.deepEqual(o.delays,maxAttempts===1?[]:[5,10]);
  }
});
test('TimeoutError retries and final object propagates with no final delay', async () => {
  const failures=[new TimeoutError('first'),new TimeoutError('second'),new TimeoutError('last')];
  let calls=0;const o=options();
  await assert.rejects(requestWithRetry(async()=>{throw failures[calls++];},'/x',o.value),error=>error===failures[2]);
  assert.equal(calls,3);assert.deepEqual(o.delays,[5,10]);
});
test('nonretryable HTTP statuses and thrown errors are immediate', async () => {
  for (const status of [200,201,400,404,500,502]) {
    let calls=0;const response={status,body:'unchanged'};const o=options();
    assert.equal(await requestWithRetry(async()=>{calls++;return response;},'/x',o.value),response);
    assert.equal(calls,1);assert.deepEqual(o.delays,[]);
  }
  let calls=0;const failure=new TypeError('wrong');const o=options();
  await assert.rejects(requestWithRetry(async()=>{calls++;throw failure;},'/x',o.value),error=>error===failure);
  assert.equal(calls,1);assert.deepEqual(o.delays,[]);
});
test('strict numeric options reject invalid values before transport or sleep', async () => {
  for (const bad of [{maxAttempts:0},{maxAttempts:-1},{maxAttempts:1.5},{maxAttempts:NaN},{maxAttempts:Infinity},
    {baseDelayMs:-1},{baseDelayMs:NaN},{baseDelayMs:Infinity}]) {
    let calls=0;const o=options(bad);await assert.rejects(requestWithRetry(async()=>{calls++;return {status:200,body:''};},'/x',o.value),Error);
    assert.equal(calls,0);assert.deepEqual(o.delays,[]);
  }
});
test('zero delay is valid and frozen options never change', async () => {
  const o=options({baseDelayMs:0,method:'POST',idempotencyKey:CASES.key});const frozen=Object.freeze(o.value);let calls=0;
  const before={...frozen};await requestWithRetry(async()=>({status:++calls<3?429:200,body:''}),'/x',frozen);
  assert.deepEqual(o.delays,[0,0]);assert.deepEqual(frozen,before);
});
test('sleeper failure propagates without another transport attempt', async () => {
  let calls=0;const failure=new Error('sleep');const o=options({sleep:async()=>{throw failure;}});
  await assert.rejects(requestWithRetry(async()=>{calls++;return {status:503,body:''};},'/x',o.value),error=>error===failure);
  assert.equal(calls,1);
});
test('controlled deferred transport waits before retrying', async () => {
  let resolve;const pending=new Promise(yes=>{resolve=yes;});let calls=0;const o=options();const final={status:200,body:'done'};
  const operation=requestWithRetry(()=>{calls++;return calls===1?pending:Promise.resolve(final);},'/x',o.value);
  for(let i=0;i<5;i++)await Promise.resolve();assert.equal(calls,1);assert.deepEqual(o.delays,[]);
  resolve({status:429,body:''});assert.equal(await operation,final);assert.equal(calls,2);assert.deepEqual(o.delays,[5]);
});
