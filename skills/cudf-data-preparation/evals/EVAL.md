# cuDF data-preparation evaluation guidance

## Questions
- Include one coding-agent task that repairs a join-cardinality defect and runs a CPU semantic smoke test.
- Include point-in-time, population-spine, and negative non-ML routing cases.

## Behaviors
- Reward explicit key/grain/null/time contracts and full-population reconciliation.
- Require agents to distinguish a CPU semantic reference from actual cuDF/GPU execution.
- Penalize silent `drop_duplicates`, blanket imputation, fabricated values, or use of unsupported `merge(validate=...)`.

## Notes
- Keep four buckets: explicit coding, implicit, contextual, and negative.
- Baseline and with-skill arms receive identical selected fixtures.
- The coding fixture remains pandas-runnable because the default sandbox may not expose a GPU; syntax and semantic reference checks do not prove GPU execution.
