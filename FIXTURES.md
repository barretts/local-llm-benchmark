# Deterministic coding and long-context fixtures

Implementation contract for the future harness. There is no frontier-model judge. Implement these fixtures, their controller-owned gold solutions and grading before benchmarking models.

## Shared construction and grading rules

- Regular fixtures: six Python and six TypeScript tasks, each run with seeds 42, 314159 and 271828, giving 36 jobs per effective configuration.
- Long fixtures: four tasks at 61,440 fully formatted input tokens, each with the same three seeds, giving 12 jobs.
- Regular gate: at least 27/36 passes. Long gate: at least 9/12. Capacity-marker probes are separate and must all pass.
- Use random.Random(seed) in the trusted fixture builder. Precompute TypeScript case data in JSON; do not use Math.random or real time in graders.
- Each fixture has source, a contract/task message, at least two public assertions and at least six held-out assertions. Property tests may add more cases. A planted bug must fail at least one public and one held-out assertion. Gold must pass every assertion for all seeds.
- Freeze/hash source, public/hidden tests, generator, oracle, contracts and seed cases before any model run. A changed grader creates a new fixture version; do not silently blend scores.
- Source folders alone are agent-writable. No edits to tests, package locks, compiler config, grading drivers or dependencies. No external services/network.
- Use Python unittest and TypeScript strict compilation plus node:test. Compile failure, missing export/signature, unauthorized action, timeout, exceeded context/token budget or any failing test means fixture failure.
- All model-written code executes in the CPU-only isolated containers specified by EXECUTION_PLAN.md. Never import it in the host controller.
- Public feedback may name the failed assertion and show public expected/actual values. Held-out tests are scored only after the final agent response; do not reveal their contents during the loop.
- Success requires complete expected test execution, not merely exit code zero or an assistant claim. Missing test-result records/counts, suppressed test execution or tampering are failures.
- Avoid sleeps for expiry/retry tests: injected clocks and sleepers record values. Async tests use controlled promises/events; wall time is only a generous deadlock guard.
- Construct genuinely buggy working implementations, not TODO/pass placeholders. The listed bug patterns are the required starting defects; do not seed extra syntax/import failures.
- Agent output and repaired diffs are archived. No human reading or LLM opinion is needed to determine passes.

## Python fixtures

### py01: expiring LRU with defensive values

Files: src/cache.py. Export TTLCache(capacity, default_ttl, clock), put(key, value, ttl=None), get(key, default=None), and __len__.

Contract:
- capacity is an integer >=1; TTL values are finite and strictly positive. Invalid values raise ValueError.
- clock() supplies monotonic numeric seconds. A value is expired when now >= expiry.
- put stores a deep copy of value, sets expiry from that operation's now, and refreshes LRU recency. Overwrite does not increase the number of keys.
- get returns a deep copy and makes an unexpired key most recently used. A miss returns the supplied default unchanged.
- Purge expired entries before capacity eviction; evict the least recently used live entry.
- len counts only live entries and purges expired ones. TTL override replaces the default for that put only.

Initial bugs: expiration uses strict greater-than, stored/returned mutable values are aliased, and expired entries can evict live ones.

Public assertions: deadline expiry; a caller mutation after put must not alter a later get.
Held-out families: get-result mutation, overwrite recency, expired-before-eviction, repeated misses, nested list/dict values, custom/default TTL, empty/live len and invalid inputs.

Oracle: a small reference OrderedDict implementation with deepcopy and the injected clock. Generate short seeded operation traces and compare results/contents to gold.

### py02: merge overlapping, not adjacent, half-open ranges

File: src/ranges.py. Export merge_ranges(ranges) -> list[tuple[int, int]].

Contract:
- Inputs are pairs of integers describing [start, end). Negative coordinates are allowed.
- end < start raises ValueError. Zero-length ranges are discarded.
- Return sorted disjoint ranges; merge actual overlap and duplicates.
- Adjacent [1,3) and [3,5) remain separate by this application's rule.
- Do not mutate the input container or its pairs. Empty input returns [].

Initial bugs: adjacency is merged and input is sorted in place.

Public assertions: adjacent ranges stay separate; unsorted input is unchanged.
Held-out families: nested/duplicate ranges, transitive overlap, negative values, zero lengths, reversed pair, empty input and stable order.

Oracle: sort a fresh normalized copy; merge only when next_start < current_end. Seeded coordinates exercise overlaps and exact boundaries.

### py03: immutable recursive configuration overlays

File: src/config.py. Export overlay_config(base: dict, overrides: dict) -> dict.

Inputs are JSON-like dictionaries with string keys and dict/list/scalar/null values.

Contract:
- Merge recursively only when both corresponding values are dicts.
- An override list replaces a base list; lists are never concatenated.
- All other override values replace the base, including False, 0, empty string and None.
- Return a fully independent deep structure; neither input changes and no mutable descendant is shared with either input.
- Keys appearing only on one side survive with independent copies.

Initial bugs: truthiness drops falsy overrides, list values concatenate, and nested dicts are mutated.

Public assertions: False/0 replace true/nonzero values; nested base remains unchanged.
Held-out families: None, empty string/list, dict-to-scalar and scalar-to-dict replacement, nested lists of dicts, missing keys, empty inputs and output mutation.

Oracle: recursive dict-only overlay plus deepcopy. Generate bounded random trees, compare equality and mutate output to check isolation.

### py04: bounded retry with correct exception policy

File: src/retry.py. Export call_with_retry(fn, *, max_attempts, base_delay, retry_on, sleep).

Contract:
- max_attempts is an integer >=1; base_delay is finite and >=0. Invalid values raise ValueError.
- Call fn immediately. Return any successful result, including None or False.
- Retry only exceptions matching the retry_on tuple. All other exceptions propagate immediately.
- Before the next attempt, call injected sleep with base_delay * 2**failure_index, where the first failure has index 0.
- Never sleep after the last allowed attempt. On final retryable failure propagate that same exception object.
- Do not catch BaseException subclasses such as KeyboardInterrupt. An exception from sleep propagates immediately.

Initial bugs: catches all ordinary exceptions and sleeps after the final failure.

Public assertions: a nonretryable ValueError is not retried; max_attempts=1 causes zero sleeps.
Held-out families: success on first/later attempt, exact exponential delay list, retry_on=(), final exception identity, falsy result, zero delay, invalid bounds and sleeper failure.

Oracle: explicit attempt loop. fn and sleep are deterministic fakes recording their call lists, with seeded success/failure schedules.

### py05: invoice money rounding across modules

Files: src/models.py, src/money.py, src/invoice.py. Preserve the LineItem frozen dataclass with unit_price: Decimal, quantity: int, discount: Decimal.

Exports:
- money.line_total(unit_price, quantity, discount) -> Decimal.
- invoice.invoice_total(items, tax_rate) -> Decimal.

Contract:
- Prices and rates are finite Decimals; price >=0; quantity integer >=0; discount/tax_rate in [0,1]. Invalid values raise ValueError.
- Each line = unit_price * quantity * (1 - discount), rounded to cents using ROUND_HALF_UP.
- Sum already-rounded lines for subtotal.
- Tax = subtotal * tax_rate, rounded separately to cents with ROUND_HALF_UP.
- Total = subtotal + rounded tax. Empty invoice yields Decimal("0.00").
- Never use binary floats or mutate input line items.

Initial bugs: bankers' rounding for line totals and rounding only the raw aggregate rather than per line.

Public assertions: one 0.005 line rounds to 0.01; two such lines total 0.02 before tax.
Held-out families: tax half-cent boundary, quantity/discount, zero and empty inputs, larger seeded decimals, invalid quantities/rates and cross-module consistency.

Oracle: Decimal operations and explicit quantize(Decimal("0.01"), rounding=ROUND_HALF_UP). Test individual helpers and the integrated invoice, not only one facade.

### py06: ordered bounded asynchronous map with cleanup

File: src/workers.py. Export async map_limited(items, worker, limit) -> list.

Contract:
- limit is an integer >=1, else ValueError. Empty input returns [] without calling worker.
- At most limit worker calls are actively executing at once.
- Output order matches input order, not completion order.
- On worker failure, cancel remaining pending/running work, await cleanup and propagate the observed exception.
- On caller cancellation, cancel/await owned worker tasks and propagate CancelledError.
- No worker owned by the call may remain running after it returns/raises.

Initial bugs: unbounded gather and returning results in completion order; no cleanup on failure.

Public assertions: deliberately reversed completion order still yields input order; active count never exceeds limit.
Held-out families: limit=1, empty input, multiple limits, worker exception cleanup, caller cancellation, deterministic blocked-worker release and invalid limit.

Oracle: semaphore plus owned tasks, ordered gather/results and explicit cancel-and-await cleanup on BaseException. Use Events and task counters, not tight timing races. Allow any genuinely first observed worker exception if two fail concurrently; tests avoid ambiguous simultaneous failures.

## TypeScript fixtures

Use strict TypeScript and readonly input types. Unless stated otherwise, invalid input must throw an Error subclass; grade behavior, not a particular error message.

### ts01: async cache coalescing, resolution-time TTL and invalidation

File: src/cache.ts. Export AsyncTTLCache<V>:
- constructor(loader: (key: string) => Promise<V>, ttlMs: number, now: () => number).
- get(key: string): Promise<V>.
- invalidate(key: string): void.

Contract:
- ttlMs finite >0. now is the injected millisecond clock.
- Concurrent gets for one key coalesce into one loader call. Promise object identity is not required.
- TTL begins when loading resolves; at now >= resolution_time + ttlMs, reload.
- Failed loads are not cached; a subsequent get tries again.
- Invalidation during an in-flight load does not reject its existing waiters, but that stale resolution must not populate the cache. A later get starts a new load.
- Keys are independent. No required cloning of generic V.

Initial bugs: rejected promises remain cached and TTL begins at request start.

Public assertions: reject then reload succeeds; a slow successful load gets its full TTL after resolution.
Held-out families: coalesced calls, exact expiry, independent keys, in-flight invalidation with out-of-order resolutions, repeated invalidation, invalid TTL and successful value reuse.

Oracle: maps for resolved entries, pending loads and per-key generation numbers; deferred promises and fake clock.

### ts02: deterministic keyset pagination

File: src/page.ts. Export RecordItem {id: string; createdAt: number}, Page and paginate(records, {limit, cursor?}).

Contract:
- Record ids are globally unique; createdAt is a finite numeric timestamp.
- Sort ascending by (createdAt, id), using deterministic string comparison (< and >), not locale-dependent collation.
- Input order is arbitrary and must not change.
- limit is integer 1..1000.
- Cursor is base64url UTF-8 JSON [createdAt, id], with strict type validation.
- Select records strictly greater than the cursor tuple, even if the cursor's original record no longer exists.
- nextCursor is encoded from the final returned item iff more records remain; otherwise null. Empty page -> null.
- Returning item object copies is not required, but source records must not be mutated.

Initial bugs: timestamp-only cursor filtering loses records sharing timestamps and input sorting mutates the array.

Public assertions: equal-timestamp records span pages without loss; original order remains unchanged.
Held-out families: whole-dataset pagination, missing cursor record, Unicode ids with deterministic comparison, malformed cursor, boundary limits, exhausted pages, unsorted inputs and no timestamp ties.

Oracle: fresh sorted copy, tuple comparison and exact cursor codec. Seeded records contain many timestamp ties; concatenate pages and compare to reference order.

### ts03: safe simultaneous UTF-16 text edits

File: src/edit.ts. Export Edit {start: number; end: number; text: string} and applyEdits(text, edits) -> string.

Contract:
- Offsets are JavaScript UTF-16 code units against the ORIGINAL string.
- Bounds are integers, 0 <= start <= end <= text.length.
- Reject a boundary between the high and low surrogate of an astral character.
- Sort a fresh edit copy. Reject overlapping edits and edits with the same start offset (including two inserts there).
- Adjacent nonoverlapping edits are valid; an insertion exactly at another edit's end is valid.
- Apply simultaneously, usually right to left. Preserve untouched CRLF and Unicode exactly.
- Do not mutate the original edit objects/array.

Initial bugs: applying left to right shifts later offsets and converting to code-point arrays interprets offsets incorrectly.

Public assertions: multiple length-changing edits use original offsets; edit after an emoji uses the correct UTF-16 offset.
Held-out families: CRLF, insertion/deletion, adjacency, overlap/same-start rejection, surrogate-split rejection, empty edits, boundaries and input immutability.

Oracle: validate original offsets, sort/validate intervals, splice in descending order. Generate disjoint edits over strings containing ASCII, CRLF and astral characters.

### ts04: stable dependency order with cycle detection

File: src/graph.ts. Export topologicalOrder(graph: Readonly<Record<string, readonly string[]>>) -> string[].

Contract:
- A key lists its prerequisite names. Include dependency names not explicitly present as keys.
- Duplicate prerequisites count once.
- At every step choose the lexicographically smallest currently-ready name, using deterministic < / > comparison.
- Every prerequisite precedes its dependent.
- A self-cycle or any cycle throws.
- Do not modify arrays/object. Treat module names as ordinary strings, including prototype-like names; use own properties and safe maps.

Initial bugs: DFS/output sorting breaks prerequisite order and missing-key dependencies are omitted.

Public assertions: a chain orders prerequisites first; an undeclared dependency is included.
Held-out families: dynamic-ready tie order, disconnected modules, duplicated edges, cycle, self-cycle, prototype-like names, empty graph and immutability.

Oracle: deduplicated prerequisite sets plus Kahn's algorithm with a sorted ready collection. Validate both exact stable result and dependency positions.

### ts05: typed immutable environment configuration

File: src/settings.ts. Export Settings {port: number; debug: boolean; tags: string[]; timeoutMs: number} and parseSettings(env, defaults) -> Settings.

Contract:
- Read only APP_PORT, APP_DEBUG, APP_TAGS and APP_TIMEOUT_MS from supplied env; never implicitly read process.env.
- undefined means use defaults. Return an independent tags array.
- Trim numeric/boolean strings. Numeric format is digits only: no signs, fractional values, exponents or empty strings. Leading zeros are allowed.
- port integer 1..65535; timeoutMs integer >=0, with zero allowed.
- debug accepts case-insensitive true/false or 1/0 only.
- Provided tags split on commas, trim, discard empties and deduplicate case-sensitively in first-occurrence order. A provided empty string clears tags.
- Do not mutate env/defaults. Reject invalid provided values.

Initial bugs: Boolean("false") is true, timeout=0 falls back to default, and default tags are returned by reference.

Public assertions: "false" disables debug; "0" preserves timeout zero.
Held-out families: " TRUE ", boundary ports, numeric syntax rejection, tags dedup/order/clear, missing values, defaults independence and invalid boolean.

Oracle: explicit parsers, range checks and fresh arrays. Seeded env maps cover valid and invalid combinations.

### ts06: atomic idempotent event store across modules

Files: src/domain.ts, src/repository.ts, src/store.ts.

Types and exports:
- Event {id: string; accountId: string; sequence: number; delta: number}.
- AccountState {balance: number; lastSequence: number}.
- Repository.read(accountId), has(eventId), transaction(fn), failNextCommit().
- Transaction.read(accountId), has(eventId), write(accountId, state), markSeen(eventId).
- EventStore(repo).apply(event) -> {duplicate: boolean; balance: number}; balance(accountId) -> number.

Contract:
- New accounts start balance=0, lastSequence=0. read returns an independent state object.
- Event ids are globally idempotent. A previously committed id returns duplicate=true and the current balance without further changes, even if repeated payload differs.
- New event sequence must be exactly the account's lastSequence+1; sequence >=1 and delta must be finite safe integers. Negative balances are allowed.
- Account update and marking event seen commit atomically; any validation/commit failure leaves both unchanged.
- failNextCommit causes exactly the next attempted commit to throw an injected failure, with full rollback. Retrying the identical event after that failure must succeed.
- Each account has its own sequence. Duplicate handling precedes new-event sequence validation.
- Do not retain/mutate borrowed state or event objects.

Initial bugs: marking seen before sequence/commit validation and repository shallow state aliasing breaks rollback.

Public assertions: invalid sequence does not poison a later valid event with that id; injected commit failure permits a successful identical retry.
Held-out families: true duplicate, duplicate changed payload, cross-account sequences, negative delta, safe-integer validation, atomic rollback, read-object mutation and event immutability.

Oracle: controller reference maps with cloned states and transactional copy/commit. Seeded event traces compare state and response flags after every event.

## Long-context fixtures: exact 64K-window work

These are single-assistant-response tasks with max 4,096 generated tokens and no intermediate tool feedback. This prevents extra read/test turns from exceeding the remaining context. The initial prompt already embeds needed source/specs. Grade after executing the final permitted tool action. Any required answer/action missing by output limit fails.

Build the same deterministic fake repository corpus family as the timing probes. Pack fully formatted input to 61,440 tokens (at most 64 below, never above). Place authoritative documents near 5%, 50% and 95% of token positions AFTER template-aware packing. Distractors are explicitly labelled obsolete, examples or unrelated tenants; the precedence rule is stated in the task. Never require guessing unstated business rules.

Use unique seed-derived document/evidence ids and names/values. All models receive identical logical seed cases, though tokenizer padding differs. Store an oracle JSON outside agent-visible paths. submit_answer has an object-valued answer and string-array evidence_ids; each fixture supplies an exact schema. Compare fields/values exactly; evidence order is source appearance order.

### long01: three-region authoritative API configuration

Task: retrieve the active route, timeout_ms and retry_limit for a named tenant/revision.

- Early authoritative document: matching tenant/revision and active route.
- Middle authoritative document: matching timeout_ms.
- Late authoritative document: matching retry_limit and confirmation of the active revision.
- Add many labelled obsolete tenant/revision snapshots with plausible conflicting values.
- The query explicitly says active matching tenant + exact revision outranks obsolete/examples; no conflicting active fact exists.

Permitted final action: one submit_answer with answer {route: string, timeout_ms: integer, retry_limit: integer} and all three evidence ids. Gold reads the generated authoritative record. Score only exact answers/evidence; incomplete retrieval fails.

Initial seed examples may use /v3/invoices, 7500ms and 4 retries, but generate alternate values per seed; do not hardcode those examples as all answers.

### long02: calculate an invoice from distant signed specifications

Task: return total as a two-decimal string, currency and schema_revision for the generated invoice.

- Early authoritative tenant manifest: currency, active revision and discount table.
- Middle authoritative invoice document: line prices, integer quantities and discount codes.
- Late authoritative policy: per-line HALF_UP cents, then separately rounded tax rate.
- Distractors contain different tenants, legacy aggregate/bankers' rounding policies and obsolete line items.
- The task explicitly names the three active documents/revision; all necessary values are present.

Permitted final action: one submit_answer with answer {total: string, currency: string, schema_revision: integer} plus all three evidence ids. Gold uses Decimal and the same per-line/tax rule as py05. Include seed cases at half-cent boundaries so using the wrong distant policy changes the answer.

### long03: repair Python settlement using distant caller/policy facts

Target: src/settlement.py only. Export settle(lines, tax_rate) -> Decimal, where each line is (Decimal unit_price, int quantity, Decimal discount).

- Early signed contract gives Decimal inputs, invalid-value policy and line formula.
- Middle repository/caller excerpt includes the full initial settlement source and call signature.
- Late policy/test excerpt requires HALF_UP per-line cents, then subtotal-tax cents, and empty=0.00.
- Add obsolete float-based implementations, wrong revision examples and unrelated modules as labelled distractors.

Permitted final action: one write_file for src/settlement.py containing complete repaired source. No edits elsewhere and no read/test turn. Initial bugs use float conversion and aggregate-only rounding. Grade using deterministic public/held-out Decimal tests derived from py05. Gold also checks the preserved callable signature and invalid inputs.

### long04: repair TypeScript retry while respecting unsafe-method policy

Target: src/request.ts only. Preserve requestWithRetry(transport, url, options) -> Promise<Response>. Embed these dependencies in the prompt and ship them read-only:

Response {status: number; body: string};
Transport = (request: {url: string; method: "GET" | "POST"; headers: Record<string,string>}) => Promise<Response>;
Options {method: "GET" | "POST"; maxAttempts: number; baseDelayMs: number; sleep: (ms: number) => Promise<void>; idempotencyKey?: string};
TimeoutError exported by src/errors.ts.

- Early active contract gives types/signature and strict numeric option validation.
- Middle active policy permits retries only for GET, or POST with a nonempty idempotencyKey. POST with that key sends "Idempotency-Key" on EVERY attempt.
- Late policy permits retry only on HTTP 429/503 or TimeoutError. Delay before retry is baseDelayMs * 2**failure_index. No delay after last attempt.
- Return a final HTTP response unchanged, even final 429/503; propagate a final TimeoutError object. Other thrown errors propagate immediately. Nonretryable POST uses one attempt.
- Fresh request/header objects per attempt; no mutation of options. Same URL/method across retries.
- maxAttempts positive integer; baseDelayMs finite >=0. A whitespace-only key is empty; a valid key is sent as originally supplied, not silently trimmed.
- Label obsolete "retry every failure" code as obsolete distractors.

Permitted final action: one write_file for src/request.ts. Initial bugs retry all errors and unsafe POST, omit the idempotency header and sleep after final failure. Grade using fake transport/deferred promises and an injected async sleeper, with tests for every active distant constraint.

## Fixture implementation acceptance before the sweep

1. Generate all seeds and save fixture-manifest.json with hashes and expected test counts.
2. Run every buggy fixture in the isolated grader and verify meaningful public + held-out failures.
3. Replace source with controller-only gold and verify all public/hidden tests pass.
4. Verify gold long answers/patches pass every oracle, including exact formatting/schema.
5. Verify random/wrong long answers, wrong evidence ids, the initial buggy long patches and a truncated-context control fail.
6. Restore buggy templates; freeze fixture version/hash and keep gold outside tool-visible roots.
7. Test source-path restrictions and ensure test-container mounts contain no personal data/secrets.
8. Only then execute model qualification. No unfinished grader, placeholder oracle or judge-model dependency is acceptable.
