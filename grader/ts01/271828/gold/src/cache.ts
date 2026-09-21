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
