'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/settings.js');
const CASES = [{"env":{"APP_PORT":" 0041906 ","APP_TIMEOUT_MS":"14592","APP_DEBUG":"0","APP_TAGS":" tag-0,X,tag-0,, X , x "},"expected":{"port":41906,"timeoutMs":14592,"debug":false,"tags":["tag-0","X","x"]}},{"env":{"APP_PORT":" 0018025 ","APP_TIMEOUT_MS":"32098","APP_DEBUG":"false","APP_TAGS":" tag-1,X,tag-1,, X , x "},"expected":{"port":18025,"timeoutMs":32098,"debug":false,"tags":["tag-1","X","x"]}},{"env":{"APP_PORT":" 0048266 ","APP_TIMEOUT_MS":"13434","APP_DEBUG":"1","APP_TAGS":" tag-2,X,tag-2,, X , x "},"expected":{"port":48266,"timeoutMs":13434,"debug":true,"tags":["tag-2","X","x"]}},{"env":{"APP_PORT":" 0058470 ","APP_TIMEOUT_MS":"71482","APP_DEBUG":"0","APP_TAGS":" tag-3,X,tag-3,, X , x "},"expected":{"port":58470,"timeoutMs":71482,"debug":false,"tags":["tag-3","X","x"]}},{"env":{"APP_PORT":" 0027652 ","APP_TIMEOUT_MS":"4165","APP_DEBUG":"false","APP_TAGS":" tag-4,X,tag-4,, X , x "},"expected":{"port":27652,"timeoutMs":4165,"debug":false,"tags":["tag-4","X","x"]}},{"env":{"APP_PORT":" 0014329 ","APP_TIMEOUT_MS":"30495","APP_DEBUG":"1","APP_TAGS":" tag-5,X,tag-5,, X , x "},"expected":{"port":14329,"timeoutMs":30495,"debug":true,"tags":["tag-5","X","x"]}},{"env":{"APP_PORT":" 001740 ","APP_TIMEOUT_MS":"73563","APP_DEBUG":"0","APP_TAGS":" tag-6,X,tag-6,, X , x "},"expected":{"port":1740,"timeoutMs":73563,"debug":false,"tags":["tag-6","X","x"]}},{"env":{"APP_PORT":" 0042591 ","APP_TIMEOUT_MS":"91924","APP_DEBUG":"TRUE","APP_TAGS":" tag-7,X,tag-7,, X , x "},"expected":{"port":42591,"timeoutMs":91924,"debug":true,"tags":["tag-7","X","x"]}},{"env":{"APP_PORT":" 0014447 ","APP_TIMEOUT_MS":"58878","APP_DEBUG":"TRUE","APP_TAGS":" tag-8,X,tag-8,, X , x "},"expected":{"port":14447,"timeoutMs":58878,"debug":true,"tags":["tag-8","X","x"]}},{"env":{"APP_PORT":" 0053047 ","APP_TIMEOUT_MS":"851","APP_DEBUG":"true","APP_TAGS":" tag-9,X,tag-9,, X , x "},"expected":{"port":53047,"timeoutMs":851,"debug":true,"tags":["tag-9","X","x"]}},{"env":{"APP_PORT":" 0045754 ","APP_TIMEOUT_MS":"55392","APP_DEBUG":"FALSE","APP_TAGS":" tag-10,X,tag-10,, X , x "},"expected":{"port":45754,"timeoutMs":55392,"debug":false,"tags":["tag-10","X","x"]}},{"env":{"APP_PORT":" 0010190 ","APP_TIMEOUT_MS":"28221","APP_DEBUG":"TRUE","APP_TAGS":" tag-11,X,tag-11,, X , x "},"expected":{"port":10190,"timeoutMs":28221,"debug":true,"tags":["tag-11","X","x"]}},{"env":{"APP_PORT":" 006699 ","APP_TIMEOUT_MS":"12156","APP_DEBUG":"false","APP_TAGS":" tag-12,X,tag-12,, X , x "},"expected":{"port":6699,"timeoutMs":12156,"debug":false,"tags":["tag-12","X","x"]}},{"env":{"APP_PORT":" 0023527 ","APP_TIMEOUT_MS":"45082","APP_DEBUG":"TRUE","APP_TAGS":" tag-13,X,tag-13,, X , x "},"expected":{"port":23527,"timeoutMs":45082,"debug":true,"tags":["tag-13","X","x"]}},{"env":{"APP_PORT":" 0052896 ","APP_TIMEOUT_MS":"5695","APP_DEBUG":"TRUE","APP_TAGS":" tag-14,X,tag-14,, X , x "},"expected":{"port":52896,"timeoutMs":5695,"debug":true,"tags":["tag-14","X","x"]}},{"env":{"APP_PORT":" 0035143 ","APP_TIMEOUT_MS":"16361","APP_DEBUG":"TRUE","APP_TAGS":" tag-15,X,tag-15,, X , x "},"expected":{"port":35143,"timeoutMs":16361,"debug":true,"tags":["tag-15","X","x"]}},{"env":{"APP_PORT":" 005165 ","APP_TIMEOUT_MS":"72357","APP_DEBUG":"0","APP_TAGS":" tag-16,X,tag-16,, X , x "},"expected":{"port":5165,"timeoutMs":72357,"debug":false,"tags":["tag-16","X","x"]}},{"env":{"APP_PORT":" 0040536 ","APP_TIMEOUT_MS":"47400","APP_DEBUG":"true","APP_TAGS":" tag-17,X,tag-17,, X , x "},"expected":{"port":40536,"timeoutMs":47400,"debug":true,"tags":["tag-17","X","x"]}},{"env":{"APP_PORT":" 0046175 ","APP_TIMEOUT_MS":"9116","APP_DEBUG":"0","APP_TAGS":" tag-18,X,tag-18,, X , x "},"expected":{"port":46175,"timeoutMs":9116,"debug":false,"tags":["tag-18","X","x"]}},{"env":{"APP_PORT":" 0014936 ","APP_TIMEOUT_MS":"37930","APP_DEBUG":"true","APP_TAGS":" tag-19,X,tag-19,, X , x "},"expected":{"port":14936,"timeoutMs":37930,"debug":true,"tags":["tag-19","X","x"]}}];
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
