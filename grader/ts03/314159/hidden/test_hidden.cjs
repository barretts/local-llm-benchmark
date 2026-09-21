'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/edit.js');
const CASES = [{"text":"head-34\r\n\ud83d\ude00abc-314159-0-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S381"},{"start":9,"end":11,"text":"E46"}],"expected":"S381-34\r\nE46abc-314159-0-\ud834\udd1etail"},{"text":"head-22\r\n\ud83d\ude00abc-314159-1-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S220"},{"start":9,"end":11,"text":"E83"}],"expected":"S220-22\r\nE83abc-314159-1-\ud834\udd1etail"},{"text":"head-49\r\n\ud83d\ude00abc-314159-2-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S721"},{"start":9,"end":11,"text":"E49"}],"expected":"S721-49\r\nE49abc-314159-2-\ud834\udd1etail"},{"text":"head-75\r\n\ud83d\ude00abc-314159-3-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S288"},{"start":9,"end":11,"text":"E31"}],"expected":"S288-75\r\nE31abc-314159-3-\ud834\udd1etail"},{"text":"head-82\r\n\ud83d\ude00abc-314159-4-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S728"},{"start":9,"end":11,"text":"E14"}],"expected":"S728-82\r\nE14abc-314159-4-\ud834\udd1etail"},{"text":"head-36\r\n\ud83d\ude00abc-314159-5-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S615"},{"start":9,"end":11,"text":"E75"}],"expected":"S615-36\r\nE75abc-314159-5-\ud834\udd1etail"},{"text":"head-85\r\n\ud83d\ude00abc-314159-6-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S258"},{"start":9,"end":11,"text":"E33"}],"expected":"S258-85\r\nE33abc-314159-6-\ud834\udd1etail"},{"text":"head-50\r\n\ud83d\ude00abc-314159-7-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S103"},{"start":9,"end":11,"text":"E74"}],"expected":"S103-50\r\nE74abc-314159-7-\ud834\udd1etail"},{"text":"head-27\r\n\ud83d\ude00abc-314159-8-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S784"},{"start":9,"end":11,"text":"E43"}],"expected":"S784-27\r\nE43abc-314159-8-\ud834\udd1etail"},{"text":"head-52\r\n\ud83d\ude00abc-314159-9-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S404"},{"start":9,"end":11,"text":"E70"}],"expected":"S404-52\r\nE70abc-314159-9-\ud834\udd1etail"},{"text":"head-52\r\n\ud83d\ude00abc-314159-10-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S815"},{"start":9,"end":11,"text":"E80"}],"expected":"S815-52\r\nE80abc-314159-10-\ud834\udd1etail"},{"text":"head-53\r\n\ud83d\ude00abc-314159-11-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S310"},{"start":9,"end":11,"text":"E21"}],"expected":"S310-53\r\nE21abc-314159-11-\ud834\udd1etail"},{"text":"head-25\r\n\ud83d\ude00abc-314159-12-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S391"},{"start":9,"end":11,"text":"E43"}],"expected":"S391-25\r\nE43abc-314159-12-\ud834\udd1etail"},{"text":"head-81\r\n\ud83d\ude00abc-314159-13-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S617"},{"start":9,"end":11,"text":"E80"}],"expected":"S617-81\r\nE80abc-314159-13-\ud834\udd1etail"},{"text":"head-88\r\n\ud83d\ude00abc-314159-14-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S341"},{"start":9,"end":11,"text":"E41"}],"expected":"S341-88\r\nE41abc-314159-14-\ud834\udd1etail"},{"text":"head-48\r\n\ud83d\ude00abc-314159-15-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S127"},{"start":9,"end":11,"text":"E83"}],"expected":"S127-48\r\nE83abc-314159-15-\ud834\udd1etail"}];
const {applyEdits} = subject;
test('seeded Unicode CRLF edits match original-offset oracle', () => {
  for (const c of CASES) assert.equal(applyEdits(c.text,c.edits),c.expected);
});
test('untouched CRLF and Unicode survive insertions and deletions exactly', () => {
  assert.equal(applyEdits('α\r\n😀\r\nZ',[{start:0,end:1,text:''},{start:7,end:7,text:'Q'}]),'\r\n😀\r\nQZ');
});
test('adjacent replacement and insertion at previous end are valid', () => {
  assert.equal(applyEdits('abcdef',[{start:0,end:2,text:'X'},{start:2,end:2,text:'Y'},{start:4,end:6,text:'Z'}]),'XYcdZ');
  assert.equal(applyEdits('abc',[{start:0,end:1,text:'A'},{start:1,end:2,text:'B'}]),'ABc');
});
test('overlap and equal starts including duplicate insertions are rejected', () => {
  for (const edits of [[{start:1,end:3,text:'x'},{start:2,end:4,text:'y'}],
    [{start:1,end:1,text:'x'},{start:1,end:1,text:'y'}],
    [{start:1,end:1,text:'x'},{start:1,end:3,text:'y'}]]) assert.throws(()=>applyEdits('abcd',edits),Error);
});
test('boundaries may not split surrogate pairs', () => {
  for (const e of [{start:2,end:2,text:'x'},{start:0,end:2,text:''},{start:2,end:4,text:'x'}])
    assert.throws(()=>applyEdits('A😀B',[e]),Error);
  assert.equal(applyEdits('A😀B',[{start:1,end:3,text:'X'}]),'AXB');
});
test('empty edits and string-edge edits work', () => {
  assert.equal(applyEdits('😀\r\n',[]),'😀\r\n'); assert.equal(applyEdits('',[]),'');
  assert.equal(applyEdits('abc',[{start:0,end:0,text:'<'},{start:3,end:3,text:'>'}]),'<abc>');
  assert.equal(applyEdits('abc',[{start:0,end:3,text:''}]),'');
});
test('integer bounds are strictly validated', () => {
  for (const [start,end] of [[-1,0],[0,4],[2,1],[0.5,1],[0,NaN],[Infinity,Infinity]])
    assert.throws(()=>applyEdits('abc',[{start,end,text:''}]),Error);
});
test('unsorted frozen edits and edit objects remain unchanged', () => {
  const edits = [{start:4,end:6,text:'!'},{start:0,end:1,text:'AAAA'}].map(Object.freeze); Object.freeze(edits);
  const before = structuredClone(edits); assert.equal(applyEdits('abcdef',edits),'AAAAbcd!'); assert.deepEqual(edits,before);
});
