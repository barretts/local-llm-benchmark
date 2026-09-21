'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/edit.js');
const CASES = [{"text":"head-91\r\n\ud83d\ude00abc-42-0-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S214"},{"start":9,"end":11,"text":"E13"}],"expected":"S214-91\r\nE13abc-42-0-\ud834\udd1etail"},{"text":"head-45\r\n\ud83d\ude00abc-42-1-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S350"},{"start":9,"end":11,"text":"E38"}],"expected":"S350-45\r\nE38abc-42-1-\ud834\udd1etail"},{"text":"head-27\r\n\ud83d\ude00abc-42-2-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S854"},{"start":9,"end":11,"text":"E23"}],"expected":"S854-27\r\nE23abc-42-2-\ud834\udd1etail"},{"text":"head-96\r\n\ud83d\ude00abc-42-3-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S858"},{"start":9,"end":11,"text":"E79"}],"expected":"S858-96\r\nE79abc-42-3-\ud834\udd1etail"},{"text":"head-21\r\n\ud83d\ude00abc-42-4-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S704"},{"start":9,"end":11,"text":"E64"}],"expected":"S704-21\r\nE64abc-42-4-\ud834\udd1etail"},{"text":"head-14\r\n\ud83d\ude00abc-42-5-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S130"},{"start":9,"end":11,"text":"E21"}],"expected":"S130-14\r\nE21abc-42-5-\ud834\udd1etail"},{"text":"head-37\r\n\ud83d\ude00abc-42-6-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S338"},{"start":9,"end":11,"text":"E74"}],"expected":"S338-37\r\nE74abc-42-6-\ud834\udd1etail"},{"text":"head-87\r\n\ud83d\ude00abc-42-7-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S127"},{"start":9,"end":11,"text":"E81"}],"expected":"S127-87\r\nE81abc-42-7-\ud834\udd1etail"},{"text":"head-35\r\n\ud83d\ude00abc-42-8-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S833"},{"start":9,"end":11,"text":"E93"}],"expected":"S833-35\r\nE93abc-42-8-\ud834\udd1etail"},{"text":"head-99\r\n\ud83d\ude00abc-42-9-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S658"},{"start":9,"end":11,"text":"E63"}],"expected":"S658-99\r\nE63abc-42-9-\ud834\udd1etail"},{"text":"head-38\r\n\ud83d\ude00abc-42-10-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S559"},{"start":9,"end":11,"text":"E85"}],"expected":"S559-38\r\nE85abc-42-10-\ud834\udd1etail"},{"text":"head-45\r\n\ud83d\ude00abc-42-11-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S928"},{"start":9,"end":11,"text":"E10"}],"expected":"S928-45\r\nE10abc-42-11-\ud834\udd1etail"},{"text":"head-30\r\n\ud83d\ude00abc-42-12-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S814"},{"start":9,"end":11,"text":"E64"}],"expected":"S814-30\r\nE64abc-42-12-\ud834\udd1etail"},{"text":"head-53\r\n\ud83d\ude00abc-42-13-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S384"},{"start":9,"end":11,"text":"E29"}],"expected":"S384-53\r\nE29abc-42-13-\ud834\udd1etail"},{"text":"head-37\r\n\ud83d\ude00abc-42-14-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S881"},{"start":9,"end":11,"text":"E53"}],"expected":"S881-37\r\nE53abc-42-14-\ud834\udd1etail"},{"text":"head-23\r\n\ud83d\ude00abc-42-15-\ud834\udd1etail","edits":[{"start":0,"end":4,"text":"S194"},{"start":9,"end":11,"text":"E58"}],"expected":"S194-23\r\nE58abc-42-15-\ud834\udd1etail"}];
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
