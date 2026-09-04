# Advanced data-science laboratories

These laboratories broaden the project without pretending that unrelated data is already available. Each has a concrete question and a promotion gate.

| Laboratory | Question | Module | Promotion evidence |
|---|---|---|---|
| Statistical inference | Are two markets' mean prices meaningfully different? | `advanced/statistics.py` | Effect, 95% CI, assumptions, multiple-test policy |
| Clustering | Which markets share level/volatility/arrival behaviour? | `advanced/clustering.py` | Stability and interpretable cluster profiles |
| NLP | Do official bulletins indicate disruption risk? | `advanced/nlp.py` | Dated corpus, labels, temporal test, explanatory terms |
| Computer vision | Can produce images help estimate quality class? | `advanced/vision.py` | Consent/licence, label protocol, group split, error audit |
| Satellite | Do vegetation indices add leading supply information? | `advanced/satellite.py` | Field/region mapping, cloud masks, acquisition-time-safe join |
| Causal inference | What is the estimated effect of a documented intervention? | `advanced/causal.py` | DAG, pre-treatment confounders, overlap, sensitivity analysis |
| Graph analytics | Which markets move together or lead others? | `advanced/graph.py` | Stable edges across rolling periods; no causal wording |
| Survival analysis | How long do price-shock episodes last? | `advanced/survival.py` | Episode definition and censoring audit |
| Optimisation | How should fixed quantity be allocated under capacity/risk? | `decision/optimization.py` | Costs, constraints, robustness scenarios |
| Sequential decisions | Can learning policies reduce market-selection regret? | `advanced/simulation.py` | Offline simulator assumptions and policy comparison |

## Important boundaries

- Correlation graphs do not show causality.
- Observational causal estimates require assumptions that cannot be proven from the table alone.
- Image models should not infer grade from unlicensed internet images.
- Satellite pixels require cloud masking and correct geography before model use.
- Simulation rewards are not real profits.
- An advanced module belongs in the main product only if it improves a predeclared metric on time-held-out real data.

