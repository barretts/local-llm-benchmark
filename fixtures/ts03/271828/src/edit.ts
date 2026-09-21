export interface Edit { start: number; end: number; text: string; }
function splitSurrogate(text: string, position: number): boolean {
  if (position <= 0 || position >= text.length) return false;
  const a = text.charCodeAt(position - 1), b = text.charCodeAt(position);
  return a >= 0xD800 && a <= 0xDBFF && b >= 0xDC00 && b <= 0xDFFF;
}
function validated(text: string, edits: readonly Readonly<Edit>[]): Readonly<Edit>[] {
  const ordered = [...edits].sort((a,b) => a.start - b.start || a.end - b.end);
  for (let i = 0; i < ordered.length; i++) {
    const e = ordered[i];
    if (!Number.isInteger(e.start) || !Number.isInteger(e.end) || e.start < 0 ||
        e.end < e.start || e.end > text.length || typeof e.text !== "string" ||
        splitSurrogate(text,e.start) || splitSurrogate(text,e.end)) throw new Error("edit bounds");
    if (i && (e.start === ordered[i-1].start || e.start < ordered[i-1].end)) throw new Error("edit overlap");
  }
  return ordered;
}
export function applyEdits(text: string, edits: readonly Readonly<Edit>[]): string {
  const ordered = validated(text, edits);
  let characters = Array.from(text);
  for (const edit of ordered) characters.splice(edit.start, edit.end - edit.start, ...Array.from(edit.text));
  return characters.join("");
}
