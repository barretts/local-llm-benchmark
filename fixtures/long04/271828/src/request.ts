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
