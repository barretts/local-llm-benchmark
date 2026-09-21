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
