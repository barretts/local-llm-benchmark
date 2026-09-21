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
export function paginate(records: readonly RecordItem[], options: Readonly<PageOptions>): Page {
  if (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 1000) throw new RangeError("limit");
  const cursor = options.cursor === undefined ? undefined : decode(options.cursor);
  const ordered = [...records].sort(compare);
  const eligible = ordered.filter(item => !cursor || compare(item, cursor) > 0);
  const items = eligible.slice(0, options.limit);
  return {items, nextCursor: eligible.length > items.length && items.length ? encode(items[items.length - 1]) : null};
}
