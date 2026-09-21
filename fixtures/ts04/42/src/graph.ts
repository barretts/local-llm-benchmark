export function topologicalOrder(graph: Readonly<Record<string, readonly string[]>>): string[] {
  const visited = new Set<string>(), active = new Set<string>(), result: string[] = [];
  function visit(name: string): void {
    if (active.has(name)) throw new Error("cycle");
    if (visited.has(name)) return;
    active.add(name);
    for (const dependency of graph[name] ?? []) {
      if (Object.prototype.hasOwnProperty.call(graph,dependency)) visit(dependency);
    }
    active.delete(name); visited.add(name); result.push(name);
  }
  for (const name of Object.keys(graph)) visit(name);
  return result.sort((a,b)=>a<b?-1:a>b?1:0);
}
