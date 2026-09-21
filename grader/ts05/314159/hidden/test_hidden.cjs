'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/settings.js');
const CASES = [{"env":{"APP_PORT":" 0012607 ","APP_TIMEOUT_MS":"36012","APP_DEBUG":"false","APP_TAGS":" tag-0,X,tag-0,, X , x "},"expected":{"port":12607,"timeoutMs":36012,"debug":false,"tags":["tag-0","X","x"]}},{"env":{"APP_PORT":" 007744 ","APP_TIMEOUT_MS":"74788","APP_DEBUG":"0","APP_TAGS":" tag-1,X,tag-1,, X , x "},"expected":{"port":7744,"timeoutMs":74788,"debug":false,"tags":["tag-1","X","x"]}},{"env":{"APP_PORT":" 0020274 ","APP_TIMEOUT_MS":"67055","APP_DEBUG":"false","APP_TAGS":" tag-2,X,tag-2,, X , x "},"expected":{"port":20274,"timeoutMs":67055,"debug":false,"tags":["tag-2","X","x"]}},{"env":{"APP_PORT":" 0037290 ","APP_TIMEOUT_MS":"80391","APP_DEBUG":"true","APP_TAGS":" tag-3,X,tag-3,, X , x "},"expected":{"port":37290,"timeoutMs":80391,"debug":true,"tags":["tag-3","X","x"]}},{"env":{"APP_PORT":" 0013464 ","APP_TIMEOUT_MS":"65975","APP_DEBUG":"1","APP_TAGS":" tag-4,X,tag-4,, X , x "},"expected":{"port":13464,"timeoutMs":65975,"debug":true,"tags":["tag-4","X","x"]}},{"env":{"APP_PORT":" 0010152 ","APP_TIMEOUT_MS":"24336","APP_DEBUG":"TRUE","APP_TAGS":" tag-5,X,tag-5,, X , x "},"expected":{"port":10152,"timeoutMs":24336,"debug":true,"tags":["tag-5","X","x"]}},{"env":{"APP_PORT":" 00251 ","APP_TIMEOUT_MS":"66127","APP_DEBUG":"0","APP_TAGS":" tag-6,X,tag-6,, X , x "},"expected":{"port":251,"timeoutMs":66127,"debug":false,"tags":["tag-6","X","x"]}},{"env":{"APP_PORT":" 0017136 ","APP_TIMEOUT_MS":"43983","APP_DEBUG":"FALSE","APP_TAGS":" tag-7,X,tag-7,, X , x "},"expected":{"port":17136,"timeoutMs":43983,"debug":false,"tags":["tag-7","X","x"]}},{"env":{"APP_PORT":" 0021733 ","APP_TIMEOUT_MS":"91591","APP_DEBUG":"TRUE","APP_TAGS":" tag-8,X,tag-8,, X , x "},"expected":{"port":21733,"timeoutMs":91591,"debug":true,"tags":["tag-8","X","x"]}},{"env":{"APP_PORT":" 0013464 ","APP_TIMEOUT_MS":"11723","APP_DEBUG":"true","APP_TAGS":" tag-9,X,tag-9,, X , x "},"expected":{"port":13464,"timeoutMs":11723,"debug":true,"tags":["tag-9","X","x"]}},{"env":{"APP_PORT":" 0018625 ","APP_TIMEOUT_MS":"34484","APP_DEBUG":"1","APP_TAGS":" tag-10,X,tag-10,, X , x "},"expected":{"port":18625,"timeoutMs":34484,"debug":true,"tags":["tag-10","X","x"]}},{"env":{"APP_PORT":" 0035980 ","APP_TIMEOUT_MS":"80818","APP_DEBUG":"0","APP_TAGS":" tag-11,X,tag-11,, X , x "},"expected":{"port":35980,"timeoutMs":80818,"debug":false,"tags":["tag-11","X","x"]}},{"env":{"APP_PORT":" 0063690 ","APP_TIMEOUT_MS":"32749","APP_DEBUG":"false","APP_TAGS":" tag-12,X,tag-12,, X , x "},"expected":{"port":63690,"timeoutMs":32749,"debug":false,"tags":["tag-12","X","x"]}},{"env":{"APP_PORT":" 0037568 ","APP_TIMEOUT_MS":"15884","APP_DEBUG":"1","APP_TAGS":" tag-13,X,tag-13,, X , x "},"expected":{"port":37568,"timeoutMs":15884,"debug":true,"tags":["tag-13","X","x"]}},{"env":{"APP_PORT":" 0043241 ","APP_TIMEOUT_MS":"13593","APP_DEBUG":"FALSE","APP_TAGS":" tag-14,X,tag-14,, X , x "},"expected":{"port":43241,"timeoutMs":13593,"debug":false,"tags":["tag-14","X","x"]}},{"env":{"APP_PORT":" 0045049 ","APP_TIMEOUT_MS":"31008","APP_DEBUG":"true","APP_TAGS":" tag-15,X,tag-15,, X , x "},"expected":{"port":45049,"timeoutMs":31008,"debug":true,"tags":["tag-15","X","x"]}},{"env":{"APP_PORT":" 007394 ","APP_TIMEOUT_MS":"18594","APP_DEBUG":"true","APP_TAGS":" tag-16,X,tag-16,, X , x "},"expected":{"port":7394,"timeoutMs":18594,"debug":true,"tags":["tag-16","X","x"]}},{"env":{"APP_PORT":" 0063044 ","APP_TIMEOUT_MS":"95958","APP_DEBUG":"TRUE","APP_TAGS":" tag-17,X,tag-17,, X , x "},"expected":{"port":63044,"timeoutMs":95958,"debug":true,"tags":["tag-17","X","x"]}},{"env":{"APP_PORT":" 0011122 ","APP_TIMEOUT_MS":"71370","APP_DEBUG":"TRUE","APP_TAGS":" tag-18,X,tag-18,, X , x "},"expected":{"port":11122,"timeoutMs":71370,"debug":true,"tags":["tag-18","X","x"]}},{"env":{"APP_PORT":" 002548 ","APP_TIMEOUT_MS":"32385","APP_DEBUG":"TRUE","APP_TAGS":" tag-19,X,tag-19,, X , x "},"expected":{"port":2548,"timeoutMs":32385,"debug":true,"tags":["tag-19","X","x"]}}];
const {parseSettings} = subject;
const defaults = {port:8080,debug:true,tags:['base'],timeoutMs:500};
test('seeded valid settings match the explicit parser oracle', () => {
  for (const c of CASES) assert.deepEqual(parseSettings(c.env,defaults),c.expected);
});
test('case-insensitive trimmed booleans accept only true false one zero', () => {
  for (const value of [' TRUE ','true',' 1 ']) assert.equal(parseSettings({APP_DEBUG:value},defaults).debug,true);
  for (const value of [' FALSE ','false',' 0 ']) assert.equal(parseSettings({APP_DEBUG:value},defaults).debug,false);
  for (const value of ['yes','no','','2','truth','Falsex']) assert.throws(()=>parseSettings({APP_DEBUG:value},defaults),Error);
});
test('port boundaries and leading zero digits are valid', () => {
  for (const [value,expected] of [['1',1],['65535',65535],[' 00042 ',42]]) assert.equal(parseSettings({APP_PORT:value},defaults).port,expected);
  for (const value of ['0','65536']) assert.throws(()=>parseSettings({APP_PORT:value},defaults),Error);
});
test('numeric syntax rejects signs fractions exponents empty and nonfinite text', () => {
  for (const key of ['APP_PORT','APP_TIMEOUT_MS']) for (const value of ['+1','-1','1.0','1e3','',' ','0x10','Infinity','NaN','1 2'])
    assert.throws(()=>parseSettings({[key]:value},defaults),Error);
});
test('tags trim deduplicate case-sensitively preserve first order and clear', () => {
  assert.deepEqual(parseSettings({APP_TAGS:' b,a,b, B , ,a,C '},defaults).tags,['b','a','B','C']);
  assert.deepEqual(parseSettings({APP_TAGS:''},defaults).tags,[]);
  assert.deepEqual(parseSettings({APP_TAGS:' , , '},defaults).tags,[]);
});
test('missing and undefined values use independent defaults', () => {
  const d = {port:9,debug:false,tags:['a'],timeoutMs:0};
  const first = parseSettings({APP_PORT:undefined,APP_DEBUG:undefined},d);
  assert.deepEqual(first,d); first.tags.push('changed'); assert.deepEqual(d.tags,['a']);
  assert.deepEqual(parseSettings({},d).tags,['a']);
});
test('env defaults and their tags are never mutated', () => {
  const d = Object.freeze({...defaults,tags:Object.freeze(['base'])});
  const env = Object.freeze({APP_TAGS:'x,x',APP_TIMEOUT_MS:'0'}); const before=structuredClone(env);
  const actual=parseSettings(env,d); assert.equal(actual.timeoutMs,0); assert.deepEqual(actual.tags,['x']);
  assert.deepEqual(env,before); assert.deepEqual(d.tags,['base']);
});
test('unrelated supplied keys cannot override defaults', () => {
  assert.deepEqual(parseSettings({PORT:'42',DEBUG:'false',APP_OTHER:'x'},defaults),defaults);
  const result=parseSettings({},defaults); assert.notEqual(result.tags,defaults.tags);
});
