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
