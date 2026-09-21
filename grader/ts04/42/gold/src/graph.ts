export function topologicalOrder(graph: Readonly<Record<string, readonly string[]>>): string[] {
  const nodes = new Set<string>(Object.keys(graph));
  for (const name of Object.keys(graph)) for (const dependency of graph[name]) nodes.add(dependency);
  const prerequisites = new Map<string, Set<string>>();
  const dependents = new Map<string, Set<string>>();
  for (const name of nodes) {
    prerequisites.set(name,new Set(Object.prototype.hasOwnProperty.call(graph,name) ? graph[name] : []));
    dependents.set(name,new Set());
  }
  for (const [name,dependencies] of prerequisites) for (const dependency of dependencies) dependents.get(dependency)!.add(name);
  const ready = [...nodes].filter(name=>prerequisites.get(name)!.size===0);
  const result: string[] = [];
  while (ready.length) {
    ready.sort((a,b)=>a<b?-1:a>b?1:0);
    const name = ready.shift()!; result.push(name);
    for (const dependent of dependents.get(name)!) {
      const dependencies = prerequisites.get(dependent)!; dependencies.delete(name);
      if (dependencies.size===0) ready.push(dependent);
    }
  }
  if (result.length!==nodes.size) throw new Error("cycle");
  return result;
}
