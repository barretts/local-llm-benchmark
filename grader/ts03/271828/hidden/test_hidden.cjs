'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/edit.js');
const CASES = [{"text":"head-97\r\n\ud83d\ude00abc-271828-0-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S927"},{"start":9,"end":11,"text":"E29"}],"expected":"S927-97\r\nE29abc-271828-0-\ud834\udd1etail"},{"text":"head-22\r\n\ud83d\ude00abc-271828-1-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S498"},{"start":9,"end":11,"text":"E15"}],"expected":"S498-22\r\nE15abc-271828-1-\ud834\udd1etail"},{"text":"head-96\r\n\ud83d\ude00abc-271828-2-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S228"},{"start":9,"end":11,"text":"E40"}],"expected":"S228-96\r\nE40abc-271828-2-\ud834\udd1etail"},{"text":"head-10\r\n\ud83d\ude00abc-271828-3-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S287"},{"start":9,"end":11,"text":"E11"}],"expected":"S287-10\r\nE11abc-271828-3-\ud834\udd1etail"},{"text":"head-79\r\n\ud83d\ude00abc-271828-4-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S319"},{"start":9,"end":11,"text":"E47"}],"expected":"S319-79\r\nE47abc-271828-4-\ud834\udd1etail"},{"text":"head-94\r\n\ud83d\ude00abc-271828-5-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S684"},{"start":9,"end":11,"text":"E58"}],"expected":"S684-94\r\nE58abc-271828-5-\ud834\udd1etail"},{"text":"head-32\r\n\ud83d\ude00abc-271828-6-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S723"},{"start":9,"end":11,"text":"E57"}],"expected":"S723-32\r\nE57abc-271828-6-\ud834\udd1etail"},{"text":"head-47\r\n\ud83d\ude00abc-271828-7-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S191"},{"start":9,"end":11,"text":"E73"}],"expected":"S191-47\r\nE73abc-271828-7-\ud834\udd1etail"},{"text":"head-82\r\n\ud83d\ude00abc-271828-8-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S232"},{"start":9,"end":11,"text":"E66"}],"expected":"S232-82\r\nE66abc-271828-8-\ud834\udd1etail"},{"text":"head-96\r\n\ud83d\ude00abc-271828-9-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S447"},{"start":9,"end":11,"text":"E52"}],"expected":"S447-96\r\nE52abc-271828-9-\ud834\udd1etail"},{"text":"head-48\r\n\ud83d\ude00abc-271828-10-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S306"},{"start":9,"end":11,"text":"E33"}],"expected":"S306-48\r\nE33abc-271828-10-\ud834\udd1etail"},{"text":"head-89\r\n\ud83d\ude00abc-271828-11-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S130"},{"start":9,"end":11,"text":"E59"}],"expected":"S130-89\r\nE59abc-271828-11-\ud834\udd1etail"},{"text":"head-94\r\n\ud83d\ude00abc-271828-12-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S672"},{"start":9,"end":11,"text":"E23"}],"expected":"S672-94\r\nE23abc-271828-12-\ud834\udd1etail"},{"text":"head-48\r\n\ud83d\ude00abc-271828-13-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S688"},{"start":9,"end":11,"text":"E84"}],"expected":"S688-48\r\nE84abc-271828-13-\ud834\udd1etail"},{"text":"head-21\r\n\ud83d\ude00abc-271828-14-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S351"},{"start":9,"end":11,"text":"E77"}],"expected":"S351-21\r\nE77abc-271828-14-\ud834\udd1etail"},{"text":"head-84\r\n\ud83d\ude00abc-271828-15-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S385"},{"start":9,"end":11,"text":"E54"}],"expected":"S385-84\r\nE54abc-271828-15-\ud834\udd1etail"}];
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
