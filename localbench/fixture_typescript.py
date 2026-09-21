"""Deterministic TypeScript fixtures; all emitted code is graded in containers.

This trusted builder never invokes Node, tsc, or imports generated source. Seeded
case data is computed here with random.Random and embedded in immutable tests.
"""

from __future__ import annotations

import json
import random
import textwrap


def _text(value: str) -> str:
    return textwrap.dedent(value).lstrip("\n").rstrip() + "\n"


def _tests(module: str, cases: object, body: str, *, extra: str = "") -> str:
    return (
        "'use strict';\nconst test = require('node:test');\n"
        "const assert = require('node:assert/strict');\n"
        f"const subject = require('/workspace/.build/{module}.js');\n"
        f"const CASES = {json.dumps(cases, ensure_ascii=True, separators=(',', ':'))};\n"
        + _text(extra)
        + _text(body)
    )


def _result(source: dict[str, str], gold: dict[str, str], contract: str,
            cases: object, public: str, hidden: str, public_count: int,
            hidden_count: int) -> dict:
    return {
        "language": "typescript", "source": source, "gold": gold,
        "public_tests": {"test_public.cjs": public},
        "hidden_tests": {"test_hidden.cjs": hidden},
        "public_count": public_count, "hidden_count": hidden_count,
        "contract": _text(contract), "cases": cases,
    }


def _cache(seed: int) -> dict:
    rng = random.Random(seed)
    cases = {"ttl": rng.randint(10, 70), "keys": [f"key-{seed}-{i}" for i in range(4)],
             "values": [rng.randint(-1000, 1000) for _ in range(4)]}
    source = _text('''
        export class AsyncTTLCache<V> {
          private entries = new Map<string, {promise: Promise<V>; expiresAt: number}>();
          constructor(private loader: (key: string) => Promise<V>, private ttlMs: number,
                      private now: () => number) {
            if (!Number.isFinite(ttlMs) || ttlMs <= 0) throw new RangeError("ttlMs");
          }
          get(key: string): Promise<V> {
            const existing = this.entries.get(key);
            const start = this.now();
            if (existing && start < existing.expiresAt) return existing.promise;
            const promise = Promise.resolve().then(() => this.loader(key));
            this.entries.set(key, {promise, expiresAt: start + this.ttlMs});
            return promise;
          }
          invalidate(key: string): void { this.entries.delete(key); }
        }
    ''')
    gold = _text('''
        export class AsyncTTLCache<V> {
          private resolved = new Map<string, {value: V; expiresAt: number}>();
          private pending = new Map<string, Promise<V>>();
          private generations = new Map<string, number>();
          constructor(private loader: (key: string) => Promise<V>, private ttlMs: number,
                      private now: () => number) {
            if (!Number.isFinite(ttlMs) || ttlMs <= 0) throw new RangeError("ttlMs");
          }
          get(key: string): Promise<V> {
            const entry = this.resolved.get(key);
            if (entry && this.now() < entry.expiresAt) return Promise.resolve(entry.value);
            this.resolved.delete(key);
            const active = this.pending.get(key);
            if (active) return active;
            const generation = this.generations.get(key) ?? 0;
            const request: Promise<V> = Promise.resolve().then(() => this.loader(key)).then(
              value => {
                if ((this.generations.get(key) ?? 0) === generation && this.pending.get(key) === request) {
                  this.pending.delete(key);
                  this.resolved.set(key, {value, expiresAt: this.now() + this.ttlMs});
                }
                return value;
              },
              error => {
                if (this.pending.get(key) === request) this.pending.delete(key);
                throw error;
              }
            );
            this.pending.set(key, request);
            return request;
          }
          invalidate(key: string): void {
            this.generations.set(key, (this.generations.get(key) ?? 0) + 1);
            this.resolved.delete(key);
            this.pending.delete(key);
          }
        }
    ''')
    extra = '''
        function deferred() {
          let resolve, reject;
          const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
          return {promise, resolve, reject};
        }
        async function tick() { for (let i = 0; i < 6; i++) await Promise.resolve(); }
        const {AsyncTTLCache} = subject;
    '''
    public = _tests("cache", cases, '''
        test('a rejected load is retried', async () => {
          let calls = 0;
          const failure = new Error('loader failure');
          const cache = new AsyncTTLCache(async () => { if (++calls === 1) throw failure; return 17; }, 10, () => 0);
          await assert.rejects(cache.get('a'), error => error === failure);
          assert.equal(await cache.get('a'), 17);
          assert.equal(calls, 2);
        });
        test('a slow load receives its full resolution-time TTL', async () => {
          let now = 0, calls = 0;
          const d = deferred();
          const cache = new AsyncTTLCache(() => { calls++; return calls === 1 ? d.promise : Promise.resolve(99); }, 10, () => now);
          const first = cache.get('a'); await tick(); now = 20; d.resolve(7);
          assert.equal(await first, 7); now = 29;
          assert.equal(await cache.get('a'), 7);
          assert.equal(calls, 1);
        });
    ''', extra=extra)
    hidden = _tests("cache", cases, '''
        test('concurrent calls coalesce until resolution and values are reused', async () => {
          const d = deferred(); let calls = 0;
          const cache = new AsyncTTLCache(() => { calls++; return d.promise; }, CASES.ttl, () => 0);
          const waiters = Array.from({length: 12}, () => cache.get(CASES.keys[0]));
          await tick(); assert.equal(calls, 1); d.resolve(CASES.values[0]);
          assert.deepEqual(await Promise.all(waiters), Array(12).fill(CASES.values[0]));
          assert.equal(await cache.get(CASES.keys[0]), CASES.values[0]); assert.equal(calls, 1);
        });
        test('exact resolution deadline expires', async () => {
          let now = 0, calls = 0;
          const cache = new AsyncTTLCache(async () => ++calls, CASES.ttl, () => now);
          assert.equal(await cache.get('a'), 1); now = CASES.ttl - 1;
          assert.equal(await cache.get('a'), 1); now = CASES.ttl;
          assert.equal(await cache.get('a'), 2);
        });
        test('failed pending load is removed, including coalesced rejection', async () => {
          let calls = 0; const d = deferred(), failure = new Error('seeded failure');
          const cache = new AsyncTTLCache(() => ++calls === 1 ? d.promise : Promise.resolve(CASES.values[1]), CASES.ttl, () => 0);
          const a = cache.get('a'), b = cache.get('a');
          const errors = Promise.allSettled([a, b]); await tick(); d.reject(failure);
          assert.deepEqual((await errors).map(x => x.reason), [failure, failure]);
          assert.equal(await cache.get('a'), CASES.values[1]); assert.equal(calls, 2);
        });
        test('keys have independent pending loads and expiries', async () => {
          let now = 0; const calls = new Map();
          const cache = new AsyncTTLCache(async key => { calls.set(key, (calls.get(key) || 0) + 1); return key; }, CASES.ttl, () => now);
          assert.deepEqual(await Promise.all(CASES.keys.map(key => cache.get(key))), CASES.keys);
          cache.invalidate(CASES.keys[1]); assert.equal(await cache.get(CASES.keys[1]), CASES.keys[1]);
          assert.equal(await cache.get(CASES.keys[0]), CASES.keys[0]);
          assert.equal(calls.get(CASES.keys[0]), 1); assert.equal(calls.get(CASES.keys[1]), 2);
        });
        test('invalidated stale loads cannot overwrite newer out-of-order resolutions', async () => {
          const old = deferred(), fresh = deferred(); let calls = 0;
          const cache = new AsyncTTLCache(() => ++calls === 1 ? old.promise : fresh.promise, CASES.ttl, () => 0);
          const oldWaiter = cache.get('x'); await tick(); cache.invalidate('x');
          const newWaiter = cache.get('x'); await tick(); fresh.resolve(22);
          assert.equal(await newWaiter, 22); old.resolve(11); assert.equal(await oldWaiter, 11);
          assert.equal(await cache.get('x'), 22); assert.equal(calls, 2);
        });
        test('repeated invalidation excludes stale values without rejecting waiters', async () => {
          const pending = [deferred(), deferred(), deferred()]; let calls = 0;
          const cache = new AsyncTTLCache(() => pending[calls++].promise, CASES.ttl, () => 0);
          const a = cache.get('x'); await tick(); cache.invalidate('x'); cache.invalidate('x');
          const b = cache.get('x'); await tick(); cache.invalidate('x');
          pending[0].resolve(1); pending[1].resolve(2); assert.deepEqual(await Promise.all([a, b]), [1, 2]);
          const c = cache.get('x'); await tick(); pending[2].resolve(3);
          assert.equal(await c, 3); assert.equal(calls, 3);
        });
        test('TTL rejects nonpositive and nonfinite values', () => {
          for (const ttl of [0, -1, NaN, Infinity, -Infinity])
            assert.throws(() => new AsyncTTLCache(async () => 1, ttl, () => 0), Error);
        });
        test('seeded slow resolutions retain full TTL on all keys', async () => {
          for (let i = 0; i < CASES.keys.length; i++) {
            let now = 0, calls = 0; const d = deferred();
            const cache = new AsyncTTLCache(() => ++calls === 1 ? d.promise : Promise.resolve(-99999), CASES.ttl, () => now);
            const p = cache.get(CASES.keys[i]); await tick(); now = CASES.ttl * 3;
            d.resolve(CASES.values[i]); await p; now += CASES.ttl - 1;
            assert.equal(await cache.get(CASES.keys[i]), CASES.values[i]); assert.equal(calls, 1);
          }
        });
    ''', extra=extra)
    return _result({"src/cache.ts": source}, {"src/cache.ts": gold}, '''
        Repair src/cache.ts. Preserve exported AsyncTTLCache<V>, constructor(loader:
        (key:string)=>Promise<V>, ttlMs:number, now:()=>number), get(key):Promise<V>
        and invalidate(key):void. TTL must be finite >0. Concurrent gets for one key
        coalesce. TTL starts on resolution and expires at now >= resolution + ttlMs.
        Failed loads are never cached. Invalidation permits existing waiters to
        resolve but their stale load cannot populate the cache; a later get starts
        a new load. Keys are independent. No generic value cloning is required.
    ''', cases, public, hidden, 2, 8)


_DOMAIN = _text('''
    export interface Event { id: string; accountId: string; sequence: number; delta: number; }
    export interface AccountState { balance: number; lastSequence: number; }
''')


def _store(seed: int) -> dict:
    rng = random.Random(seed)
    balances = {"alpha": 0, "beta": 0, "constructor": 0}
    sequences = {name: 0 for name in balances}
    events = []
    for i in range(24):
        account = rng.choice(list(balances))
        delta = rng.randint(-200,200)
        sequences[account] += 1
        balances[account] += delta
        event = {"id": f"event-{seed}-{i}", "accountId":account,
                 "sequence":sequences[account], "delta":delta}
        events.append({"event": event, "expected":{"duplicate":False,"balance":balances[account]}})
    cases = {"trace":events,"balances":balances,"sequences":sequences}
    repository = _text('''
        import {AccountState} from "./domain";
        export class Transaction {
          constructor(private states: Map<string,AccountState>, private seen: Set<string>) {}
          read(accountId: string): AccountState { return this.states.get(accountId) ?? {balance:0,lastSequence:0}; }
          has(eventId: string): boolean { return this.seen.has(eventId); }
          write(accountId: string, state: AccountState): void { this.states.set(accountId,state); }
          markSeen(eventId: string): void { this.seen.add(eventId); }
        }
        export class Repository {
          private states = new Map<string,AccountState>();
          private seen = new Set<string>();
          private fail = false;
          read(accountId: string): AccountState { return this.states.get(accountId) ?? {balance:0,lastSequence:0}; }
          has(eventId: string): boolean { return this.seen.has(eventId); }
          failNextCommit(): void { this.fail = true; }
          transaction<T>(fn: (tx: Transaction) => T): T {
            const states = new Map(this.states);
            const result = fn(new Transaction(states,this.seen));
            if (this.fail) { this.fail=false; throw new Error("injected commit failure"); }
            this.states = states;
            return result;
          }
        }
    ''')
    gold_repository = _text('''
        import {AccountState} from "./domain";
        export class Transaction {
          constructor(private states: Map<string,AccountState>, private seen: Set<string>) {}
          read(accountId: string): AccountState {
            const state = this.states.get(accountId);
            return state ? {...state} : {balance:0,lastSequence:0};
          }
          has(eventId: string): boolean { return this.seen.has(eventId); }
          write(accountId: string, state: AccountState): void { this.states.set(accountId,{...state}); }
          markSeen(eventId: string): void { this.seen.add(eventId); }
        }
        export class Repository {
          private states = new Map<string,AccountState>();
          private seen = new Set<string>();
          private fail = false;
          read(accountId: string): AccountState {
            const state = this.states.get(accountId);
            return state ? {...state} : {balance:0,lastSequence:0};
          }
          has(eventId: string): boolean { return this.seen.has(eventId); }
          failNextCommit(): void { this.fail = true; }
          transaction<T>(fn: (tx: Transaction) => T): T {
            const states = new Map([...this.states].map(([id,state])=>[id,{...state}]));
            const seen = new Set(this.seen);
            const result = fn(new Transaction(states,seen));
            if (this.fail) { this.fail=false; throw new Error("injected commit failure"); }
            this.states = states; this.seen = seen;
            return result;
          }
        }
    ''')
    store_header = _text('''
        import {Event} from "./domain";
        import {Repository} from "./repository";
        export class EventStore {
          constructor(private repo: Repository) {}
          balance(accountId: string): number { return this.repo.read(accountId).balance; }
          apply(event: Readonly<Event>): {duplicate: boolean; balance: number} {
            if (this.repo.has(event.id)) return {duplicate:true,balance:this.balance(event.accountId)};
            return this.repo.transaction(tx => {
    ''')
    store_tail = _text('''
            });
          }
        }
    ''')
    source_store = store_header + _text('''
        tx.markSeen(event.id);
        const state = tx.read(event.accountId);
        if (!Number.isSafeInteger(event.sequence) || event.sequence < 1 ||
            !Number.isSafeInteger(event.delta) || event.sequence !== state.lastSequence + 1) throw new Error("invalid event");
        state.balance += event.delta; state.lastSequence = event.sequence;
        tx.write(event.accountId,state);
        return {duplicate:false,balance:state.balance};
    ''') + store_tail
    gold_store = store_header + _text('''
        const previous = tx.read(event.accountId);
        if (!Number.isSafeInteger(event.sequence) || event.sequence < 1 ||
            !Number.isSafeInteger(event.delta) || event.sequence !== previous.lastSequence + 1) throw new Error("invalid event");
        const state = {balance:previous.balance + event.delta,lastSequence:event.sequence};
        tx.write(event.accountId,state); tx.markSeen(event.id);
        return {duplicate:false,balance:state.balance};
    ''') + store_tail
    extra = '''
        const {EventStore} = subject;
        const {Repository} = require('/workspace/.build/repository.js');
        function fresh() { const repo = new Repository(); return {repo,store:new EventStore(repo)}; }
        const event = (id,accountId,sequence,delta) => ({id,accountId,sequence,delta});
    '''
    public = _tests("store", cases, '''
        test('invalid sequence does not poison a later valid event id', () => {
          const {repo,store}=fresh(); assert.throws(()=>store.apply(event('a','x',2,5)),Error);
          assert.equal(repo.has('a'),false);
          assert.deepEqual(store.apply(event('a','x',1,5)),{duplicate:false,balance:5});
        });
        test('injected commit failure permits identical retry', () => {
          const {repo,store}=fresh(); repo.failNextCommit(); const e=event('a','x',1,7);
          assert.throws(()=>store.apply(e),Error); assert.equal(store.balance('x'),0); assert.equal(repo.has('a'),false);
          assert.deepEqual(store.apply(e),{duplicate:false,balance:7});
        });
    ''', extra=extra)
    hidden = _tests("store", cases, '''
        test('seeded cross-account event trace matches controller state after every event', () => {
          const {repo,store}=fresh();
          for (const row of CASES.trace) {
            const before=structuredClone(row.event); assert.deepEqual(store.apply(row.event),row.expected);
            assert.deepEqual(row.event,before); assert.equal(repo.has(row.event.id),true);
          }
          for (const [id,balance] of Object.entries(CASES.balances)) {
            assert.equal(store.balance(id),balance); assert.deepEqual(repo.read(id),{balance,lastSequence:CASES.sequences[id]});
          }
        });
        test('true duplicate returns current balance without reapplying', () => {
          const {store}=fresh(); const e=event('one','a',1,10); store.apply(e); store.apply(event('two','a',2,3));
          assert.deepEqual(store.apply(e),{duplicate:true,balance:13}); assert.equal(store.balance('a'),13);
        });
        test('globally duplicate changed payload precedes sequence validation', () => {
          const {store}=fresh(); store.apply(event('same','a',1,8)); store.apply(event('other','b',1,-3));
          assert.deepEqual(store.apply(event('same','b',NaN,Infinity)),{duplicate:true,balance:-3});
          assert.equal(store.balance('a'),8); assert.equal(store.balance('b'),-3);
        });
        test('each account has independent sequences and negative balances are valid', () => {
          const {store}=fresh(); assert.deepEqual(store.apply(event('a','x',1,-10)),{duplicate:false,balance:-10});
          assert.deepEqual(store.apply(event('b','y',1,2)),{duplicate:false,balance:2});
          assert.throws(()=>store.apply(event('c','x',3,1)),Error);
          assert.deepEqual(store.apply(event('c','x',2,-5)),{duplicate:false,balance:-15});
        });
        test('safe integer and sequence validation leave ids and accounts unmodified', () => {
          for (const [sequence,delta] of [[0,1],[-1,1],[1.5,1],[NaN,1],[Infinity,1],[1,0.5],[1,NaN],[1,Infinity],[1,Number.MAX_SAFE_INTEGER+1],[Number.MAX_SAFE_INTEGER+1,1]]) {
            const {repo,store}=fresh(); assert.throws(()=>store.apply(event('invalid','a',sequence,delta)),Error);
            assert.equal(repo.has('invalid'),false); assert.deepEqual(repo.read('a'),{balance:0,lastSequence:0});
          }
        });
        test('existing account commit failure rolls back state and seen together', () => {
          const {repo,store}=fresh(); store.apply(event('first','a',1,11)); repo.failNextCommit(); const e=event('next','a',2,-7);
          assert.throws(()=>store.apply(e),Error); assert.deepEqual(repo.read('a'),{balance:11,lastSequence:1}); assert.equal(repo.has('next'),false);
          assert.deepEqual(store.apply(e),{duplicate:false,balance:4});
        });
        test('repository read and transaction write never retain borrowed state', () => {
          const {repo,store}=fresh(); store.apply(event('first','a',1,5)); const state=repo.read('a'); state.balance=900; state.lastSequence=90;
          assert.deepEqual(repo.read('a'),{balance:5,lastSequence:1});
          const borrowed={balance:8,lastSequence:2}; repo.transaction(tx=>tx.write('b',borrowed)); borrowed.balance=99;
          assert.deepEqual(repo.read('b'),{balance:8,lastSequence:2});
          repo.transaction(tx=>{const r=tx.read('b');r.balance=444;}); assert.equal(repo.read('b').balance,8);
        });
        test('transaction callback failure rolls back every write and seen marker', () => {
          const {repo}=fresh(); const failure=new Error('callback');
          assert.throws(()=>repo.transaction(tx=>{tx.write('a',{balance:5,lastSequence:1});tx.markSeen('e');throw failure;}),error=>error===failure);
          assert.equal(repo.has('e'),false); assert.deepEqual(repo.read('a'),{balance:0,lastSequence:0});
        });
        test('failNextCommit affects one attempted commit and validation does not consume it', () => {
          const {repo,store}=fresh(); repo.failNextCommit(); assert.throws(()=>store.apply(event('bad','a',2,3)),Error);
          assert.throws(()=>store.apply(event('good','a',1,3)),Error); assert.equal(repo.has('good'),false);
          assert.deepEqual(store.apply(event('good','a',1,3)),{duplicate:false,balance:3});
          assert.deepEqual(store.apply(event('next','a',2,1)),{duplicate:false,balance:4});
        });
        test('frozen events stay unchanged and unknown account reads are independent', () => {
          const {repo,store}=fresh(); const e=Object.freeze(event('frozen','a',1,4)); const before=structuredClone(e);
          assert.deepEqual(store.apply(e),{duplicate:false,balance:4}); assert.deepEqual(e,before);
          const r=repo.read('missing'); r.balance=42; assert.deepEqual(repo.read('missing'),{balance:0,lastSequence:0});
        });
    ''', extra=extra)
    return _result({"src/domain.ts":_DOMAIN,"src/repository.ts":repository,"src/store.ts":source_store},
                   {"src/domain.ts":_DOMAIN,"src/repository.ts":gold_repository,"src/store.ts":gold_store}, '''
        Repair src/domain.ts, src/repository.ts and src/store.ts. Preserve Event
        {id:string;accountId:string;sequence:number;delta:number}, AccountState
        {balance:number;lastSequence:number}, Repository.read/has/transaction/
        failNextCommit, Transaction.read/has/write/markSeen, and EventStore(repo).
        apply(event):{duplicate:boolean;balance:number} and balance(accountId).
        Unknown accounts start at 0/0; reads are independent. Event ids are
        globally idempotent: a committed duplicate returns true and the payload's
        account's current balance without changes, even if payload differs.
        Duplicate handling precedes new-event validation. New sequence is exactly
        account.lastSequence+1; sequence>=1 and sequence/delta are finite safe
        integers. Negative balances are allowed. State and seen id commit
        atomically; validation/callback/commit failures roll back both.
        failNextCommit makes exactly the next attempted commit throw, and retrying
        its identical event succeeds. Each account sequences independently.
        Never retain/mutate borrowed state or event objects. transaction(fn) is
        synchronous and returns fn's result after a successful commit.
    ''', cases, public, hidden, 2, 10)


_PAGE_HEADER = _text('''
    declare const Buffer: {
      from(value: string, encoding?: string): {toString(encoding?: string): string};
    };
    export interface RecordItem { id: string; createdAt: number; }
    export interface Page { items: RecordItem[]; nextCursor: string | null; }
    export interface PageOptions { limit: number; cursor?: string; }
    function compare(a: RecordItem, b: RecordItem): number {
      return a.createdAt - b.createdAt || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
    }
    function encode(item: RecordItem): string {
      return Buffer.from(JSON.stringify([item.createdAt, item.id]), "utf8").toString("base64url");
    }
    function decode(cursor: string): RecordItem {
      if (typeof cursor !== "string" || !/^[A-Za-z0-9_-]+$/.test(cursor)) throw new Error("cursor");
      const bytes = Buffer.from(cursor, "base64url");
      if (bytes.toString("base64url") !== cursor) throw new Error("cursor");
      const text = bytes.toString("utf8");
      if (Buffer.from(text, "utf8").toString("base64url") !== cursor) throw new Error("cursor UTF-8");
      const tuple: unknown = JSON.parse(text);
      if (!Array.isArray(tuple) || tuple.length !== 2 || typeof tuple[0] !== "number" ||
          !Number.isFinite(tuple[0]) || typeof tuple[1] !== "string") throw new Error("cursor tuple");
      return {createdAt: tuple[0], id: tuple[1]};
    }
''')


def _page(seed: int) -> dict:
    rng = random.Random(seed)
    records = [{"id": f"r-{seed}-{i:02}", "createdAt": rng.randint(-3, 5)} for i in range(29)]
    rng.shuffle(records)
    cases = {"records": records, "ordered": sorted(records, key=lambda x: (x["createdAt"], x["id"])),
             "limits": [1, 2, 4, 7, 1000]}
    source = _PAGE_HEADER + _text('''
        export function paginate(records: readonly RecordItem[], options: Readonly<PageOptions>): Page {
          if (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 1000) throw new RangeError("limit");
          const cursor = options.cursor === undefined ? undefined : decode(options.cursor);
          const ordered = (records as RecordItem[]).sort(compare);
          const eligible = ordered.filter(item => !cursor || item.createdAt > cursor.createdAt);
          const items = eligible.slice(0, options.limit);
          return {items, nextCursor: eligible.length > items.length && items.length ? encode(items[items.length - 1]) : null};
        }
    ''')
    gold = _PAGE_HEADER + _text('''
        export function paginate(records: readonly RecordItem[], options: Readonly<PageOptions>): Page {
          if (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 1000) throw new RangeError("limit");
          const cursor = options.cursor === undefined ? undefined : decode(options.cursor);
          const ordered = [...records].sort(compare);
          const eligible = ordered.filter(item => !cursor || compare(item, cursor) > 0);
          const items = eligible.slice(0, options.limit);
          return {items, nextCursor: eligible.length > items.length && items.length ? encode(items[items.length - 1]) : null};
        }
    ''')
    extra = '''
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
    '''
    public = _tests("page", cases, '''
        test('timestamp ties span pages without losing ids', () => {
          const records = [{id:'c',createdAt:1},{id:'a',createdAt:1},{id:'b',createdAt:1}];
          assert.deepEqual(allPages(records, 1).map(x => x.id), ['a','b','c']);
        });
        test('input order remains unchanged', () => {
          const records = [{id:'b',createdAt:2},{id:'a',createdAt:1}];
          const before = structuredClone(records); paginate(records, {limit:1}); assert.deepEqual(records, before);
        });
    ''', extra=extra)
    hidden = _tests("page", cases, '''
        test('seeded complete pagination equals the tuple oracle at all limits', () => {
          for (const limit of CASES.limits) assert.deepEqual(allPages(structuredClone(CASES.records), limit), CASES.ordered);
        });
        test('a deleted cursor record still filters by its tuple', () => {
          const data = [{id:'a',createdAt:2},{id:'c',createdAt:2},{id:'a',createdAt:3}];
          assert.deepEqual(paginate(data, {limit:10,cursor:encode(2,'b')}).items, [data[1],data[2]]);
        });
        test('Unicode ids use deterministic code-unit ordering and UTF-8 cursors', () => {
          const ids = ['é','Z','a','😀','ä','A','中','𝄞'];
          const data = ids.map(id => ({id, createdAt:9}));
          const expected = [...ids].sort((a,b)=>a<b?-1:a>b?1:0);
          assert.deepEqual(allPages(data, 1).map(x=>x.id), expected);
          const p = paginate(data, {limit:3});
          assert.equal(p.nextCursor, encode(9,expected[2]));
        });
        test('malformed cursor encodings and invalid tuple types throw', () => {
          const wrong = ['', '!', 'a=', 'a', '____', encode(1, 'a') + '='];
          for (const value of [null, {}, [1], [1,'a',3], ['1','a'], [1,3], [null,'a']])
            wrong.push(Buffer.from(JSON.stringify(value)).toString('base64url'));
          for (const cursor of wrong) assert.throws(()=>paginate([], {limit:1,cursor}), Error);
        });
        test('limit endpoints work and invalid limits fail', () => {
          assert.equal(paginate(CASES.records, {limit:1}).items.length, 1);
          assert.equal(paginate(CASES.records, {limit:1000}).items.length, CASES.records.length);
          for (const limit of [0,-1,1001,1.5,NaN,Infinity]) assert.throws(()=>paginate([], {limit}), Error);
        });
        test('exhausted and empty pages have null cursors', () => {
          assert.deepEqual(paginate([], {limit:2}), {items:[],nextCursor:null});
          assert.deepEqual(paginate(CASES.records, {limit:2,cursor:encode(99999,'last')}), {items:[],nextCursor:null});
          const last = CASES.ordered[CASES.ordered.length-1];
          assert.deepEqual(paginate(CASES.records, {limit:1,cursor:encode(last.createdAt,last.id)}), {items:[],nextCursor:null});
        });
        test('frozen unsorted source records are never changed', () => {
          const records = CASES.records.map(x=>Object.freeze({...x})); Object.freeze(records);
          const before = structuredClone(records); paginate(records, {limit:5}); assert.deepEqual(records,before);
        });
        test('distinct timestamps use exact last-item cursor and no extra cursor', () => {
          const data = [{id:'x',createdAt:7},{id:'z',createdAt:-1},{id:'y',createdAt:3}];
          const p = paginate(data,{limit:2}); assert.deepEqual(p.items,[data[1],data[2]]);
          assert.equal(p.nextCursor,encode(3,'y'));
          assert.deepEqual(paginate(data,{limit:2,cursor:p.nextCursor}),{items:[data[0]],nextCursor:null});
        });
    ''', extra=extra)
    return _result({"src/page.ts": source}, {"src/page.ts": gold}, '''
        Repair src/page.ts and preserve RecordItem {id:string;createdAt:number}, Page
        {items:RecordItem[];nextCursor:string|null}, and paginate(records,
        {limit,cursor?}). Records have unique ids and finite timestamps. Sort by
        ascending (createdAt,id), comparing strings with < and >. Do not mutate
        readonly input. limit is an integer 1..1000. A cursor is base64url UTF-8
        JSON [finite number,string] with strict validation. Select tuples strictly
        greater than the cursor even after its record was deleted. nextCursor is
        encoded from the final returned item exactly when more remain; otherwise
        null, including an empty page. Copying item objects is not required.
    ''', cases, public, hidden, 2, 8)


_EDIT_HEADER = _text('''
    export interface Edit { start: number; end: number; text: string; }
    function splitSurrogate(text: string, position: number): boolean {
      if (position <= 0 || position >= text.length) return false;
      const a = text.charCodeAt(position - 1), b = text.charCodeAt(position);
      return a >= 0xD800 && a <= 0xDBFF && b >= 0xDC00 && b <= 0xDFFF;
    }
    function validated(text: string, edits: readonly Readonly<Edit>[]): Readonly<Edit>[] {
      const ordered = [...edits].sort((a,b) => a.start - b.start || a.end - b.end);
      for (let i = 0; i < ordered.length; i++) {
        const e = ordered[i];
        if (!Number.isInteger(e.start) || !Number.isInteger(e.end) || e.start < 0 ||
            e.end < e.start || e.end > text.length || typeof e.text !== "string" ||
            splitSurrogate(text,e.start) || splitSurrogate(text,e.end)) throw new Error("edit bounds");
        if (i && (e.start === ordered[i-1].start || e.start < ordered[i-1].end)) throw new Error("edit overlap");
      }
      return ordered;
    }
''')


def _edits(seed: int) -> dict:
    rng = random.Random(seed)
    cases = []
    for i in range(16):
        text = f"head-{rng.randint(10,99)}\r\n😀abc-{seed}-{i}-𝄞tail"
        edits = [{"start": 0, "end": 4, "text": f"S{rng.randint(100,999)}"},
                 {"start": 9, "end": 11, "text": f"E{rng.randint(10,99)}"}]
        # Offsets are UTF-16, so the emoji after 9 code units occupies [9,11).
        units = text.encode("utf-16-le")
        out = units
        for e in reversed(edits):
            out = out[:2*e["start"]] + e["text"].encode("utf-16-le") + out[2*e["end"]:]
        cases.append({"text": text, "edits": edits, "expected": out.decode("utf-16-le")})
    source = _EDIT_HEADER + _text('''
        export function applyEdits(text: string, edits: readonly Readonly<Edit>[]): string {
          const ordered = validated(text, edits);
          let characters = Array.from(text);
          for (const edit of ordered) characters.splice(edit.start, edit.end - edit.start, ...Array.from(edit.text));
          return characters.join("");
        }
    ''')
    gold = _EDIT_HEADER + _text('''
        export function applyEdits(text: string, edits: readonly Readonly<Edit>[]): string {
          const ordered = validated(text, edits);
          let result = text;
          for (let i = ordered.length - 1; i >= 0; i--) {
            const edit = ordered[i];
            result = result.slice(0, edit.start) + edit.text + result.slice(edit.end);
          }
          return result;
        }
    ''')
    extra = "const {applyEdits} = subject;"
    public = _tests("edit", cases, r'''
        test('simultaneous length-changing edits use original offsets', () => {
          assert.equal(applyEdits('abcdef', [{start:0,end:1,text:'AAAA'},{start:4,end:6,text:'!'}]), 'AAAAbcd!');
        });
        test('offset after emoji counts UTF-16 code units', () => {
          assert.equal(applyEdits('A😀BCD', [{start:3,end:4,text:'x'}]), 'A😀xCD');
        });
    ''', extra=extra)
    hidden = _tests("edit", cases, r'''
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
    ''', extra=extra)
    return _result({"src/edit.ts": source}, {"src/edit.ts": gold}, '''
        Repair src/edit.ts, preserving Edit {start:number;end:number;text:string}
        and applyEdits(text,readonly edits):string. Offsets are UTF-16 code units
        in the ORIGINAL string. Bounds must be integers 0<=start<=end<=length and
        cannot split an astral surrogate pair. Sort a fresh copy; reject overlaps
        and equal starts, including duplicate inserts. Adjacent edits and an
        insertion at another edit's end are valid. Apply simultaneously, preserve
        untouched CRLF/Unicode exactly, and never mutate edit objects/array.
    ''', cases, public, hidden, 2, 8)


def _topology(graph: dict[str, list[str]]) -> list[str]:
    nodes = set(graph)
    for dependencies in graph.values():
        nodes.update(dependencies)
    remaining = {node: set(graph.get(node, [])) for node in nodes}
    output = []
    while remaining:
        ready = sorted(node for node, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise ValueError("cycle")
        node = ready[0]
        output.append(node)
        del remaining[node]
        for dependencies in remaining.values():
            dependencies.discard(node)
    return output


def _graph(seed: int) -> dict:
    rng = random.Random(seed)
    cases = []
    for i in range(14):
        names = [f"module-{seed}-{j:02}" for j in range(8)]
        rng.shuffle(names)
        graph = {name: rng.sample(names[:j], rng.randint(0,min(j,3))) for j,name in enumerate(names)}
        graph[names[-1]].append(f"external-{i}")
        cases.append({"graph": graph, "expected": _topology(graph)})
    source = _text('''
        export function topologicalOrder(graph: Readonly<Record<string, readonly string[]>>): string[] {
          const visited = new Set<string>(), active = new Set<string>(), result: string[] = [];
          function visit(name: string): void {
            if (active.has(name)) throw new Error("cycle");
            if (visited.has(name)) return;
            active.add(name);
            for (const dependency of graph[name] ?? []) {
              if (Object.prototype.hasOwnProperty.call(graph,dependency)) visit(dependency);
            }
            active.delete(name); visited.add(name); result.push(name);
          }
          for (const name of Object.keys(graph)) visit(name);
          return result.sort((a,b)=>a<b?-1:a>b?1:0);
        }
    ''')
    gold = _text('''
        export function topologicalOrder(graph: Readonly<Record<string, readonly string[]>>): string[] {
          const nodes = new Set<string>(Object.keys(graph));
          for (const name of Object.keys(graph)) for (const dependency of graph[name]) nodes.add(dependency);
          const prerequisites = new Map<string, Set<string>>();
          const dependents = new Map<string, Set<string>>();
          for (const name of nodes) {
            prerequisites.set(name,new Set(Object.prototype.hasOwnProperty.call(graph,name) ? graph[name] : []));
            dependents.set(name,new Set());
          }
          for (const [name,dependencies] of prerequisites) for (const dependency of dependencies) dependents.get(dependency)!.add(name);
          const ready = [...nodes].filter(name=>prerequisites.get(name)!.size===0);
          const result: string[] = [];
          while (ready.length) {
            ready.sort((a,b)=>a<b?-1:a>b?1:0);
            const name = ready.shift()!; result.push(name);
            for (const dependent of dependents.get(name)!) {
              const dependencies = prerequisites.get(dependent)!; dependencies.delete(name);
              if (dependencies.size===0) ready.push(dependent);
            }
          }
          if (result.length!==nodes.size) throw new Error("cycle");
          return result;
        }
    ''')
    extra = "const {topologicalOrder} = subject;"
    public = _tests("graph", cases, '''
        test('a lexical reverse chain orders prerequisites first', () => {
          assert.deepEqual(topologicalOrder({a:['b'],b:['c'],c:[]}),['c','b','a']);
        });
        test('an undeclared dependency is included', () => {
          assert.deepEqual(topologicalOrder({app:['external']}),['external','app']);
        });
    ''', extra=extra)
    hidden = _tests("graph", cases, '''
        test('seeded DAGs match stable Kahn oracle including missing nodes', () => {
          for (const c of CASES) assert.deepEqual(topologicalOrder(c.graph),c.expected);
        });
        test('lexical minimum is chosen anew when a module becomes ready', () => {
          assert.deepEqual(topologicalOrder({a:['b'],b:[],c:[],d:['a','c']}),['b','a','c','d']);
        });
        test('disconnected components and duplicated edges are handled', () => {
          assert.deepEqual(topologicalOrder({b:['z','z'],z:[],a:[],c:['a']}),['a','c','z','b']);
        });
        test('multi-node cycles throw', () => {
          assert.throws(()=>topologicalOrder({a:['b'],b:['c'],c:['a']}),Error);
          assert.throws(()=>topologicalOrder({ready:[],a:['b'],b:['a']}),Error);
        });
        test('self cycles throw', () => { assert.throws(()=>topologicalOrder({x:['x']}),Error); });
        test('prototype-like names and own properties are ordinary module names', () => {
          const graph = JSON.parse('{"__proto__":["constructor"],"constructor":["toString"],"toString":[]}');
          assert.deepEqual(topologicalOrder(graph),['toString','constructor','__proto__']);
          const inherited = Object.create({bad:['bad']}); inherited.app=['dependency'];
          assert.deepEqual(topologicalOrder(inherited),['dependency','app']);
        });
        test('empty graph returns empty output', () => { assert.deepEqual(topologicalOrder({}),[]); });
        test('frozen source object and prerequisite arrays remain unchanged', () => {
          const graph = {a:Object.freeze(['z','z']),z:Object.freeze([])}; Object.freeze(graph);
          const before = structuredClone(graph); assert.deepEqual(topologicalOrder(graph),['z','a']); assert.deepEqual(graph,before);
        });
    ''', extra=extra)
    return _result({"src/graph.ts": source}, {"src/graph.ts": gold}, '''
        Repair src/graph.ts and preserve topologicalOrder(graph:
        Readonly<Record<string,readonly string[]>>):string[]. A key lists its
        prerequisites. Include undeclared dependency names; deduplicate edges.
        At every step choose the lexically smallest CURRENTLY ready name with
        deterministic < / > comparison. Prerequisites precede dependents. Any
        cycle including self-cycle throws. Do not mutate input. Use own properties
        and safe maps so prototype-like names are ordinary names.
    ''', cases, public, hidden, 2, 8)


_SETTINGS_HEADER = _text('''
    export interface Settings { port: number; debug: boolean; tags: string[]; timeoutMs: number; }
    type Defaults = {readonly port: number; readonly debug: boolean; readonly tags: readonly string[]; readonly timeoutMs: number};
    function numeric(value: string, minimum: number, maximum: number): number {
      const text = value.trim();
      if (!/^[0-9]+$/.test(text)) throw new Error("numeric syntax");
      const number = Number(text);
      if (!Number.isInteger(number) || !Number.isFinite(number) || number < minimum || number > maximum) throw new RangeError("numeric range");
      return number;
    }
    function booleanText(value: string): string {
      const text = value.trim().toLowerCase();
      if (!["true","false","1","0"].includes(text)) throw new Error("boolean syntax");
      return text;
    }
    function tags(value: string): string[] { return [...new Set(value.split(",").map(x=>x.trim()).filter(x=>x!==""))]; }
''')


def _settings(seed: int) -> dict:
    rng = random.Random(seed)
    cases = []
    for i in range(20):
        port, timeout, debug = rng.randint(1,65535), rng.randint(0,100000), bool(rng.getrandbits(1))
        env = {"APP_PORT": f" 00{port} ", "APP_TIMEOUT_MS": str(timeout),
               "APP_DEBUG": rng.choice(["true","TRUE","1"]) if debug else rng.choice(["false","FALSE","0"]),
               "APP_TAGS": f" tag-{i},X,tag-{i},, X , x "}
        cases.append({"env":env,"expected":{"port":port,"timeoutMs":timeout,"debug":debug,"tags":[f"tag-{i}","X","x"]}})
    source = _SETTINGS_HEADER + _text('''
        export function parseSettings(env: Readonly<Record<string,string | undefined>>, defaults: Defaults): Settings {
          return {
            port: env.APP_PORT === undefined ? defaults.port : numeric(env.APP_PORT,1,65535),
            debug: env.APP_DEBUG === undefined ? defaults.debug : Boolean(booleanText(env.APP_DEBUG)),
            tags: env.APP_TAGS === undefined ? defaults.tags as string[] : tags(env.APP_TAGS),
            timeoutMs: env.APP_TIMEOUT_MS === undefined ? defaults.timeoutMs : numeric(env.APP_TIMEOUT_MS,0,Infinity) || defaults.timeoutMs
          };
        }
    ''')
    gold = _SETTINGS_HEADER + _text('''
        export function parseSettings(env: Readonly<Record<string,string | undefined>>, defaults: Defaults): Settings {
          const debugText = env.APP_DEBUG === undefined ? undefined : booleanText(env.APP_DEBUG);
          return {
            port: env.APP_PORT === undefined ? defaults.port : numeric(env.APP_PORT,1,65535),
            debug: debugText === undefined ? defaults.debug : debugText === "true" || debugText === "1",
            tags: env.APP_TAGS === undefined ? [...defaults.tags] : tags(env.APP_TAGS),
            timeoutMs: env.APP_TIMEOUT_MS === undefined ? defaults.timeoutMs : numeric(env.APP_TIMEOUT_MS,0,Infinity)
          };
        }
    ''')
    extra = "const {parseSettings} = subject;\nconst defaults = {port:8080,debug:true,tags:['base'],timeoutMs:500};"
    public = _tests("settings", cases, '''
        test('false text disables debug', () => { assert.equal(parseSettings({APP_DEBUG:'false'},defaults).debug,false); });
        test('zero timeout is preserved', () => { assert.equal(parseSettings({APP_TIMEOUT_MS:'0'},defaults).timeoutMs,0); });
    ''', extra=extra)
    hidden = _tests("settings", cases, '''
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
    ''', extra=extra)
    return _result({"src/settings.ts": source}, {"src/settings.ts": gold}, '''
        Repair src/settings.ts. Preserve Settings {port:number;debug:boolean;
        tags:string[];timeoutMs:number} and parseSettings(env,defaults):Settings.
        Read only supplied APP_PORT, APP_DEBUG, APP_TAGS, APP_TIMEOUT_MS; never
        process.env. undefined uses defaults, with an independent tags array.
        Trim numeric/boolean strings. Numbers must be digits only (leading zeros
        allowed), port 1..65535 and integer timeout>=0. No signs/fractions/exponents/
        empty text. Debug accepts case-insensitive true/false or 1/0 only. Provided
        tags split on commas, trim, discard empties and deduplicate case-sensitively
        in first order. Empty tags clear. Reject invalid values; mutate no inputs.
    ''', cases, public, hidden, 2, 8)


_REQUEST_HEADER = _text('''
    import {TimeoutError} from "./errors";
    export interface Response { status: number; body: string; }
    export type Transport = (request: {url: string; method: "GET" | "POST"; headers: Record<string,string>}) => Promise<Response>;
    export interface Options {
      method: "GET" | "POST"; maxAttempts: number; baseDelayMs: number;
      sleep: (ms: number) => Promise<void>; idempotencyKey?: string;
    }
    function validate(options: Readonly<Options>): void {
      if (!Number.isInteger(options.maxAttempts) || options.maxAttempts < 1 ||
          !Number.isFinite(options.baseDelayMs) || options.baseDelayMs < 0) throw new RangeError("retry options");
    }
''')


def _request(seed: int) -> dict:
    rng = random.Random(seed)
    cases = {"delay":rng.randint(0,25),"url":f"/tenant-{seed}/events",
             "key":f"  event-key-{seed}-{rng.randint(1000,9999)}  ",
             "statuses":[rng.choice([429,503]) for _ in range(3)]+[200],
             "body":f"response-{rng.randint(10000,99999)}"}
    errors = _text('''
        export class TimeoutError extends Error {
          constructor(message: string = "timeout") { super(message); this.name = "TimeoutError"; }
        }
    ''')
    source = _REQUEST_HEADER + _text('''
        export async function requestWithRetry(transport: Transport, url: string, options: Readonly<Options>): Promise<Response> {
          validate(options);
          let lastResponse: Response | undefined, lastError: unknown;
          for (let attempt=0; attempt<options.maxAttempts; attempt++) {
            try {
              const response = await transport({url,method:options.method,headers:{}});
              if (response.status !== 429 && response.status !== 503) return response;
              lastResponse = response; lastError = undefined;
            } catch (error) { lastError = error; lastResponse = undefined; }
            await options.sleep(options.baseDelayMs * 2 ** attempt);
          }
          if (lastResponse) return lastResponse;
          throw lastError;
        }
    ''')
    gold = _REQUEST_HEADER + _text('''
        export async function requestWithRetry(transport: Transport, url: string, options: Readonly<Options>): Promise<Response> {
          validate(options);
          const key = options.idempotencyKey;
          const keyedPost = options.method === "POST" && key !== undefined && key.trim().length > 0;
          const retryAllowed = options.method === "GET" || keyedPost;
          const attempts = retryAllowed ? options.maxAttempts : 1;
          for (let attempt=0; attempt<attempts; attempt++) {
            const headers: Record<string,string> = {};
            if (keyedPost) headers["Idempotency-Key"] = key!;
            let response: Response;
            try { response = await transport({url,method:options.method,headers}); }
            catch (error) {
              if (!(error instanceof TimeoutError) || attempt === attempts - 1) throw error;
              await options.sleep(options.baseDelayMs * 2 ** attempt);
              continue;
            }
            if ((response.status !== 429 && response.status !== 503) || attempt === attempts - 1) return response;
            await options.sleep(options.baseDelayMs * 2 ** attempt);
          }
          throw new Error("unreachable");
        }
    ''')
    extra = '''
        const {requestWithRetry} = subject;
        const {TimeoutError} = require('/workspace/.build/errors.js');
        function options(overrides={}) { const delays=[]; return {delays,value:{method:'GET',maxAttempts:3,baseDelayMs:5,sleep:async ms=>{delays.push(ms);},...overrides}}; }
    '''
    public = _tests("request", cases, '''
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
    ''', extra=extra)
    hidden = _tests("request", cases, r'''
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
    ''', extra=extra)
    contract = '''
        Repair ONLY src/request.ts in one final write_file with complete source.
        Preserve requestWithRetry(transport,url,options):Promise<Response> and
        exported Response {status:number;body:string}, Transport taking
        {url:string;method:"GET"|"POST";headers:Record<string,string>}, Options
        {method:"GET"|"POST";maxAttempts:number;baseDelayMs:number;
        sleep:(ms:number)=>Promise<void>;idempotencyKey?:string}.
        src/errors.ts exports TimeoutError and is a READ-ONLY dependency.
        Active signed early contract: maxAttempts positive integer; baseDelayMs
        finite >=0. Validate before transport, including unsafe POST.
        Active signed middle policy: retry only GET or POST with a nonempty
        idempotencyKey; whitespace-only is empty. POST with a valid key sends
        Idempotency-Key on EVERY attempt, exactly as supplied without trimming.
        Active signed late policy: retry only HTTP 429/503 or TimeoutError; other
        thrown errors propagate immediately. Sleep before the next attempt with
        baseDelayMs*2**failure_index (first failure index 0), never after last.
        Return final HTTP response unchanged, including final 429/503. Propagate
        the final TimeoutError object. Unsafe POST uses one attempt. Each attempt
        gets fresh request/header objects with same URL/method. Mutate no options.
        Active contracts outrank explicitly obsolete/example/unrelated material.
    '''
    return _result({"src/request.ts":source,"src/errors.ts":errors},
                   {"src/request.ts":gold,"src/errors.ts":errors},contract,cases,public,hidden,2,10)


_BUILDERS = {"ts01":_cache,"ts02":_page,"ts03":_edits,"ts04":_graph,
             "ts05":_settings,"ts06":_store,"long04":_request}


def build(fixture_id: str, seed: int) -> dict:
    """Return deterministic, JSON-compatible source/gold/tests/contracts/cases."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    try:
        builder = _BUILDERS[fixture_id]
    except KeyError as error:
        raise ValueError(f"Unknown TypeScript fixture: {fixture_id}") from error
    return builder(seed)
