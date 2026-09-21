import {Event} from "./domain";
import {Repository} from "./repository";
export class EventStore {
  constructor(private repo: Repository) {}
  balance(accountId: string): number { return this.repo.read(accountId).balance; }
  apply(event: Readonly<Event>): {duplicate: boolean; balance: number} {
    if (this.repo.has(event.id)) return {duplicate:true,balance:this.balance(event.accountId)};
    return this.repo.transaction(tx => {
const previous = tx.read(event.accountId);
if (!Number.isSafeInteger(event.sequence) || event.sequence < 1 ||
    !Number.isSafeInteger(event.delta) || event.sequence !== previous.lastSequence + 1) throw new Error("invalid event");
const state = {balance:previous.balance + event.delta,lastSequence:event.sequence};
tx.write(event.accountId,state); tx.markSeen(event.id);
return {duplicate:false,balance:state.balance};
    });
  }
}
