'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const subject = require('/workspace/.build/settings.js');
const CASES = [{"env":{"APP_PORT":" 0044866 ","APP_TIMEOUT_MS":"20017","APP_DEBUG":"true","APP_TAGS":" tag-0,X,tag-0,, X , x "},"expected":{"port":44866,"timeoutMs":20017,"debug":true,"tags":["tag-0","X","x"]}},{"env":{"APP_PORT":" 0025490 ","APP_TIMEOUT_MS":"5843","APP_DEBUG":"true","APP_TAGS":" tag-1,X,tag-1,, X , x "},"expected":{"port":25490,"timeoutMs":5843,"debug":true,"tags":["tag-1","X","x"]}},{"env":{"APP_PORT":" 0055707 ","APP_TIMEOUT_MS":"31332","APP_DEBUG":"false","APP_TAGS":" tag-2,X,tag-2,, X , x "},"expected":{"port":55707,"timeoutMs":31332,"debug":false,"tags":["tag-2","X","x"]}},{"env":{"APP_PORT":" 00861 ","APP_TIMEOUT_MS":"71678","APP_DEBUG":"FALSE","APP_TAGS":" tag-3,X,tag-3,, X , x "},"expected":{"port":861,"timeoutMs":71678,"debug":false,"tags":["tag-3","X","x"]}},{"env":{"APP_PORT":" 0057097 ","APP_TIMEOUT_MS":"86601","APP_DEBUG":"1","APP_TAGS":" tag-4,X,tag-4,, X , x "},"expected":{"port":57097,"timeoutMs":86601,"debug":true,"tags":["tag-4","X","x"]}},{"env":{"APP_PORT":" 0025021 ","APP_TIMEOUT_MS":"99273","APP_DEBUG":"0","APP_TAGS":" tag-5,X,tag-5,, X , x "},"expected":{"port":25021,"timeoutMs":99273,"debug":false,"tags":["tag-5","X","x"]}},{"env":{"APP_PORT":" 0062542 ","APP_TIMEOUT_MS":"48192","APP_DEBUG":"false","APP_TAGS":" tag-6,X,tag-6,, X , x "},"expected":{"port":62542,"timeoutMs":48192,"debug":false,"tags":["tag-6","X","x"]}},{"env":{"APP_PORT":" 0061631 ","APP_TIMEOUT_MS":"64853","APP_DEBUG":"1","APP_TAGS":" tag-7,X,tag-7,, X , x "},"expected":{"port":61631,"timeoutMs":64853,"debug":true,"tags":["tag-7","X","x"]}},{"env":{"APP_PORT":" 008458 ","APP_TIMEOUT_MS":"57456","APP_DEBUG":"TRUE","APP_TAGS":" tag-8,X,tag-8,, X , x "},"expected":{"port":8458,"timeoutMs":57456,"debug":true,"tags":["tag-8","X","x"]}},{"env":{"APP_PORT":" 0021986 ","APP_TIMEOUT_MS":"39326","APP_DEBUG":"false","APP_TAGS":" tag-9,X,tag-9,, X , x "},"expected":{"port":21986,"timeoutMs":39326,"debug":false,"tags":["tag-9","X","x"]}},{"env":{"APP_PORT":" 0040646 ","APP_TIMEOUT_MS":"3884","APP_DEBUG":"0","APP_TAGS":" tag-10,X,tag-10,, X , x "},"expected":{"port":40646,"timeoutMs":3884,"debug":false,"tags":["tag-10","X","x"]}},{"env":{"APP_PORT":" 0036632 ","APP_TIMEOUT_MS":"13480","APP_DEBUG":"0","APP_TAGS":" tag-11,X,tag-11,, X , x "},"expected":{"port":36632,"timeoutMs":13480,"debug":false,"tags":["tag-11","X","x"]}},{"env":{"APP_PORT":" 0038056 ","APP_TIMEOUT_MS":"12211","APP_DEBUG":"0","APP_TAGS":" tag-12,X,tag-12,, X , x "},"expected":{"port":38056,"timeoutMs":12211,"debug":false,"tags":["tag-12","X","x"]}},{"env":{"APP_PORT":" 0037913 ","APP_TIMEOUT_MS":"36590","APP_DEBUG":"TRUE","APP_TAGS":" tag-13,X,tag-13,, X , x "},"expected":{"port":37913,"timeoutMs":36590,"debug":true,"tags":["tag-13","X","x"]}},{"env":{"APP_PORT":" 0039933 ","APP_TIMEOUT_MS":"28384","APP_DEBUG":"false","APP_TAGS":" tag-14,X,tag-14,, X , x "},"expected":{"port":39933,"timeoutMs":28384,"debug":false,"tags":["tag-14","X","x"]}},{"env":{"APP_PORT":" 0060687 ","APP_TIMEOUT_MS":"44555","APP_DEBUG":"0","APP_TAGS":" tag-15,X,tag-15,, X , x "},"expected":{"port":60687,"timeoutMs":44555,"debug":false,"tags":["tag-15","X","x"]}},{"env":{"APP_PORT":" 0049057 ","APP_TIMEOUT_MS":"49950","APP_DEBUG":"false","APP_TAGS":" tag-16,X,tag-16,, X , x "},"expected":{"port":49057,"timeoutMs":49950,"debug":false,"tags":["tag-16","X","x"]}},{"env":{"APP_PORT":" 0037699 ","APP_TIMEOUT_MS":"86656","APP_DEBUG":"0","APP_TAGS":" tag-17,X,tag-17,, X , x "},"expected":{"port":37699,"timeoutMs":86656,"debug":false,"tags":["tag-17","X","x"]}},{"env":{"APP_PORT":" 0060415 ","APP_TIMEOUT_MS":"92203","APP_DEBUG":"TRUE","APP_TAGS":" tag-18,X,tag-18,, X , x "},"expected":{"port":60415,"timeoutMs":92203,"debug":true,"tags":["tag-18","X","x"]}},{"env":{"APP_PORT":" 0021811 ","APP_TIMEOUT_MS":"66753","APP_DEBUG":"TRUE","APP_TAGS":" tag-19,X,tag-19,, X , x "},"expected":{"port":21811,"timeoutMs":66753,"debug":true,"tags":["tag-19","X","x"]}}];
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
