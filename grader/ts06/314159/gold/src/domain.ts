export interface Event { id: string; accountId: string; sequence: number; delta: number; }
export interface AccountState { balance: number; lastSequence: number; }
