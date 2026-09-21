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
