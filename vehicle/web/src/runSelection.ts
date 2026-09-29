type Selection = { runId: string | null; pendingCount: number | null };
const KEY = 'vehicle-run-selection';

export function readRunSelection(storage?: Pick<Storage, 'getItem'>): Selection {
  try {
    const value = JSON.parse((storage || sessionStorage).getItem(KEY) || 'null');
    if (value && typeof value === 'object') {
      if (typeof value.runId === 'string' && /^[a-f0-9]{32}$/.test(value.runId) && value.pendingCount === null) return value;
      if (value.runId === null && Number.isInteger(value.pendingCount) && value.pendingCount > 0 && value.pendingCount <= 2000) return value;
    }
  } catch { /* An unavailable browser store must not disable search. */ }
  return { runId: null, pendingCount: null };
}

export function saveRunSelection(value: Selection, storage?: Pick<Storage, 'setItem'>) {
  try { (storage || sessionStorage).setItem(KEY, JSON.stringify(value)); } catch { /* Keep current React state without browser persistence. */ }
}
