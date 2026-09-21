export class TimeoutError extends Error {
  constructor(message: string = "timeout") { super(message); this.name = "TimeoutError"; }
}
