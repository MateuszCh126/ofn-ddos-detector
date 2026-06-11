# AI Agent Instructions

This repository has a local nodesify-graphify knowledge graph. Use it before
raw file search when you need architecture, ownership, dependency, or cross-file
context.

## Graphify Files

- Graph report: `.graphify/graph_report.md`
- Interactive HTML graph: `.graphify/graph.html`
- Spread-out HTML graph: `.graphify/graph_spread.html`
- Strongly spread HTML graph: `.graphify/graph_exploded.html`
- JSON graph: `.graphify/graph.json`
- SQLite graph database: `.graphify/db.sqlite`
- Ignore rules: `.graphifyignore`

Local repository path:

```text
D:\github\github\ofn-ddos-detector
```

## Required Workflow

1. Check graph freshness:

```powershell
nodesify-graphify status --graph .
```

2. For architecture or codebase questions, read `.graphify/graph_report.md`
   first, then query the graph:

```powershell
nodesify-graphify query "architecture overview detector dashboard ofn" --budget 3000
nodesify-graphify explain "ddos_ofn"
nodesify-graphify path "run.py" "ddos_ofn"
```

3. Use normal file search only after the graph has narrowed the relevant files
   or when the task is a simple literal lookup.

4. After code changes, update the graph:

```powershell
nodesify-graphify update .
```

5. If `.graphify/graph.json` is missing or clearly stale, rebuild it:

```powershell
nodesify-graphify run .
```

## Notes

- `data/`, `artifacts/`, caches, runtime files, and binary/generated files are
  excluded by `.graphifyignore`.
- Prefer `.graphify/graph_exploded.html` for manual visual inspection when the
  default graph is too dense.
- Do not remove or overwrite `.graphify/` unless explicitly asked.
