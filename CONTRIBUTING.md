# Contributing

New recipes, ports to other GPUs or engines, and reproductions (matching or not) are all welcome.

- **Results:** open an issue with the "Recipe problem or different results" template. Include what [AGENTS.md](AGENTS.md)
  lists under "Reporting results".
- **Pull requests:** keep each one to one recipe or fix. A speed change needs before/after numbers measured back to back
  with unique prompts, and the quality suite (`correctness/eval.py --reasoning on`) for both sides.
- **No model changes hidden in a speedup.** If a change alters weights, routing or sampling, say so in the PR title.
- Contributions are under Apache-2.0, like the rest of the repo.
