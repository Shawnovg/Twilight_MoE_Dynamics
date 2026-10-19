# Structural component comparison

Reviewed 2026-10-16. Scope: components in the named formulations, not performance and not all possible extensions. MoE is a general routing-and-experts pattern; FFN-MoE is a specific application. MHA is a component, while Transformer/UT/SUT are architectures. No means absent from the formulation, not incompatible.

| Structure | Attention organization | Routed FFN experts? | Unit selected by router | Reuse across depth | Adaptive stopping / skipping |
|---|---|---|---|---|---|
| Standard MHA component | Multiple heads; all evaluated | Not part of this component | No head/expert router | Not specified by component | Not specified |
| Generic MoE component | Not required; depends on expert design | Only if experts are FFNs | Expert function | Not specified | Not intrinsic |
| Vanilla Transformer | Standard dense MHA | No; dense FFN | None | Different block parameters by layer | Fixed depth |
| Typical FFN-MoE Transformer (e.g. Switch) | Standard attention alongside MoE | Yes, at designated FFN sublayers | FFN expert | Usually separate pools per layer | No inherent ACT |
| UT | Standard MHA in repeated block | No | No head/FFN router | Same block parameters reused | Optional ACT variant |
| MoA (Zhang et al.) | Mixture of individual attention heads | No additional FFN-MoE in the described model | Individual head, with paired query/output projections | No UT-style recurrence introduced | No ACT introduced |
| MoH (Jin et al.) | Always-active shared heads plus routed heads | Not introduced by MoH | Individual routed head | No UT-style recurrence introduced | No ACT introduced |
| SUT | MoMHA: H heads per attention expert | Yes | Attention expert containing H heads; separately FFN expert | Same SUT block / expert pools reused | Stick-breaking halting |
| MoM (Gong et al.) | Pool of whole MHA modules; multiple heads per module | Yes, routed FFN-module pool | MHA module and FFN module through separate routers | Modules reusable across assembly steps | Optional SKIP; fixed assembly budget, not SUT ACT |

## Distinctions to preserve

- Vanilla attention softmax selects/weights token information; it is not a sparse router over expert networks. Standard MHA is not FFN-MoE.
- MoA has multiple attention heads overall, but each attention expert in its equations is a single head, not a nested MHA block. Shared K/V; expert-specific Q and output projection. Figure 2 depicts two aligned projection expert banks, not an added FFN-MoE. Some MoA settings activate all heads (k=E); sparsity requires k<E.
- SUT explicitly generalizes the routed attention unit to H heads per expert. Its shared K/V set has H heads, shared across E experts. Its FFN mixture is separately routed.
- MoH shared heads means always-active heads; it does not mean MoA's single shared K/V projection. KV sharing in a host such as grouped-query attention is a separate property.
- MoM is module-level selection. Attention assembly combines selected projection parameters; do not depict it as necessarily a weighted sum of independently evaluated complete MHA outputs. FFN assembly is weighted output aggregation. SKIP at one step does not imply permanent halting. Routers use GRUs to retain assembly history.
- MoH and MoM are 2024 papers, later than SUT (2023); comparisons, not predecessors of SUT.

## Sources and useful original figures

- Vanilla Transformer, Section 3 and Figure 1: https://arxiv.org/html/1706.03762
- FFN-MoE example, Switch Transformer: https://arxiv.org/html/2101.03961
- UT, Sections 2.1-2.2 and Appendix A Figure 4 / Appendix C: https://arxiv.org/html/1807.03819v3
- MoA, Sections 3-4, Equations 6-12, Figures 1-2: https://arxiv.org/html/2210.05144 . Figure 2 is the detailed Q/K/V and paired expert-projection candidate for slides.
- MoH, Sections 3.2 and 5: https://arxiv.org/html/2410.11842
- SUT, Sections 3.1-3.2 and Figure 3: https://aclanthology.org/2023.emnlp-main.12.pdf . Also checked supplied local PDF page 3.
- MoM, Sections 3.1-3.3 and 3.5, Figures 1-2: https://arxiv.org/html/2407.06677 . Figure 2 compares structural computation paths and is useful for the user's cross-architecture perspective.

## Presentation update

Add a dedicated state-aggregation slide after the introduction to dynamic halting: UT recursive interpolation versus SUT expected halted state, with common notation and active-query / halt-aware-key-value distinction. This is separate from depth recurrence: UT is recurrent because it repeatedly applies a shared block, even without halting. Do not imply interpolation creates UT recurrence. Retain compact ACT comparison on the preceding halting slide provisionally.

Use this complete table as the reading/reference master. A presentation version should be shortened for legibility; no requirement to pack all cells onto a single slide. User requests structural inclusion and compatibility, not performance rankings. Broad synthesis: routing granularity (FFN/head/multi-head expert/whole module), reuse across depth, and adaptive computation are distinct design dimensions, not a proven equivalence among architectures.
