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
export function parseSettings(env: Readonly<Record<string,string | undefined>>, defaults: Defaults): Settings {
  return {
    port: env.APP_PORT === undefined ? defaults.port : numeric(env.APP_PORT,1,65535),
    debug: env.APP_DEBUG === undefined ? defaults.debug : Boolean(booleanText(env.APP_DEBUG)),
    tags: env.APP_TAGS === undefined ? defaults.tags as string[] : tags(env.APP_TAGS),
    timeoutMs: env.APP_TIMEOUT_MS === undefined ? defaults.timeoutMs : numeric(env.APP_TIMEOUT_MS,0,Infinity) || defaults.timeoutMs
  };
}
