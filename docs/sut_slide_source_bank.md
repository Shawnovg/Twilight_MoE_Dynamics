# Original-source material for the SUT presentation

## Scope correction from the user

The presentation is about Sparse Universal Transformer. Most reading questions are personal learning details, not slide requirements. This bank is optional background material, not a proposed slide outline or a checklist to include.

Prioritize SUT's motivation, architecture, sparse feed-forward and attention computation, halting contribution, experimental evidence, limitations, and relevance to the user's structural experiments. Include only enough UT and task/dataset background to make those points understandable. Prefer original cited figures.

Explicit user requirement: original Universal Transformers Appendix A Figure 4 must occupy an important position in the deck. Use it as the architectural reference, then explain SUT by locating changes on that structure. This requirement overrides treating all UT details as optional background. Pair it with original SUT Figure 3 (PDF p.4 / proceedings p.172), whose left panel shows a sparse block and right panel shows halting-state flow. Do not claim SUT lacks an architecture diagram.

Explicit user requirement from Section 3.1 reading: show the actual sparse routing/computation flow and the configuration tuples, not only generic MoE boxes. Feed-forward experts: (E,k,D). MoMHA: (E,k,H,D,W), where H is heads per expert and D is dimension per head. Highlight E expert-specific sets of query projections and one shared set of H key/value head projections; router-weighted combination of selected expert outputs. E counts multihead experts, not total individual heads. k counts selected experts. Explain token query vectors versus learned projection parameters: tokens have distinct activations and routing scores, not their own independently learned projection matrices. Do not assert that the router takes projected Q rows merely because attention uses Q; Section 3.1 describes g(e|h). Show residual/normalization placement from SUT Figure 3 rather than assuming the original UT ordering is unchanged.

W specifies a relative-position embedding window: 2W+1 offsets from -W through 0 to +W. Do not infer a hard attention mask or a particular out-of-range clipping scheme solely from this tuple description.

Architecture comparison sequence: establish the shared-block UT structure using Figure 4; identify replacement of dense feed-forward computation by routed feed-forward experts and multihead attention by MoMHA; then highlight stick-breaking halting and its effect on attention's state inputs. Auxiliary routing/ACT losses are training objectives, not extra forward-pass blocks. Preserve distinction between the conceptual UT scaffold and exact SUT implementation, including positional encoding differences; avoid implying every decoder attention sublayer is modified identically without implementation verification.

Keep sinusoidal encoding derivations, detailed ACT code analysis, circuit-complexity results, ALBERT ablation details, and Takase-Kiyono sharing schemes out of the main deck unless later discussion establishes a direct need. Preserve those explanations in the reading journal; do not automatically convert questions into slides or backup slides.

Use original cited figures/tables where suitable, per user preference. This is a source selection record, not the final slide deck. Avoid the superseded custom CFQ schematic.

## Task and dataset explanation

- Keysers et al., CFQ paper, Figure 1 (PDF page 4): https://arxiv.org/pdf/1912.09713#page=4 . Shows rule-based generation of a natural-language question, intermediate logical form, and SPARQL query. For the semantic-parsing benchmark, input is question text and prediction target is query text; intermediate logical form is generator machinery, not a required model prediction.
- Same paper, Appendix A: https://arxiv.org/html/1912.09713v2#A1 . Actual dataset item includes question variants, SPARQL variants, answer, and metadata. Useful backup slide when explaining what a training pair contains.
- Same paper, Figure 2 and Table 4: https://arxiv.org/html/1912.09713v2 . Generalization versus compound divergence and random/MCD results. Preserve original settings and distinguish these baseline UT results from SUT's later implementation.

## Weight sharing and its trade-offs

- Original UT, Figure 2 and Appendix A Figure 4: https://arxiv.org/html/1807.03819v3 . Shared recurrent encoder/decoder structure and detailed block. Previously inspected text; obtain original figure for deck during slide assembly.
- ALBERT, Section 4.5 Table 4: https://arxiv.org/html/1909.11942#S4.SS5 . Direct parameter-sharing comparison. With embedding size E=128, all-shared: 12M parameters, average score 80.1; not-shared: 89M, average 81.6. This average combines different downstream metrics; do not call it a single-task accuracy. E=768 comparisons show the same overall direction.
- ALBERT, Table 2: https://arxiv.org/html/1909.11942 . Wider variants can exceed the BERT baseline; this is a separate scaling result, not a contradiction of fixed-width sharing ablations.
- SUT Figure 1, Table 1, Figure 3: local supplied paper. Candidate main architecture/cost visuals; preserve equal-parameter assumptions and distinguish MACs from parameter counts.

## Computational limitations: claims and scope

## Partial layer sharing: Takase and Kiyono

- Original Figure 1 and Section 2: https://arxiv.org/html/2104.06022 . Three independent parameter sets used across six layers: sequence = A A B B C C; cycle = A B C A B C; cycle (rev) = A B C C B A. In the general reversed-cycle scheme only the final cycle is reversed. A/B/C here denote trainable blocks, not data entities. Prefer original Figure 1 for the presentation.
- Main rationale: interpolate between a single reused block and fully independent layers; allocate a parameter budget across several smaller blocks instead of a single wide repeated block. Static assignment by depth, not input-conditioned MoE routing. Retrieved HTML is revised v4 (2023) of the paper initially posted in 2021.

- Hahn (2020): https://aclanthology.org/2020.tacl-1.11/ . Hard-attention limitations on parity/hierarchy; soft-attention results concern sensitivity and cross-entropy under assumptions, not a blanket impossibility of correct classification at every length.
- Hao, Angluin & Frank (2022): https://arxiv.org/abs/2204.06618 . Unique-hard-attention models bounded by AC0, excluding parity and Dyck recognition; averaging-hard attention differs and can recognize Dyck-1. Keep attention variants explicit.
- Merrill, Sabharwal & Smith (2022): https://aclanthology.org/2022.tacl-1.49/ . Saturated attention with floating-point values admits a TC0 upper bound. This is a stronger class than AC0; do not infer parity is excluded by this result. Deeper-class separations may require unproved complexity assumptions.
- Liu et al. (2022): https://arxiv.org/abs/2210.10749 . Constructive O(log T)-depth automaton simulations, with constant-depth shortcuts for many cases; learned shortcuts can be brittle. Not solely a negative expressivity result.
- Dziri et al. (2023): https://arxiv.org/abs/2305.18654 . Empirical compositional failures on multiplication, logic-grid, and dynamic-programming tasks; distinguish experimental observations from universal impossibility theorems.

Slide framing: fixed computational depth and learned shortcuts can limit systematic extrapolation. Tied weights permit repeated computation, but do not guarantee learning the correct algorithm or unlimited generalization at fixed inference depth.

## Updated presentation direction (2026-10-16)

User explicitly requests an intelligently organized architecture-first narrative, not literal transcription of every reading question. This is provisional planning, not authorization to interrupt reading and build the deck now.

Suggested methods sequence:
1. UT foundation: original UT Figure 4 prominently; shared refinement block and adaptive depth.
2. SUT overview: Figure 3 left as map of modifications; introduce sparse FFN experts (E,k,D) as the simple routing primitive before the more complex attention extension. Keep this together on one slide if legible.
3. Mixture of attention: dedicated slide on shared K/V projections, multiple expert-specific Q projections, top-k routing and weighted outputs; preserve distinction between expert and attention head. Inspect original MoA paper (user's spoken IMOA likely MoA, arxiv2210.05144) equations and available original figures during preparation, and verify SUT's MoMHA differences. Do not claim the figure selection has already been completed.
4. Mutual-information maximization: dedicated slide with -H(E)+H(E|H), balanced marginal routing and confident per-token decisions, minimal illustrative example. Resolve paper/code sign issues before claiming reproduction; avoid exporting all entropy tutorials into slides.
5. Stick-breaking halting plus compact ACT comparison on one slide, common notation for UT and SUT. UT adds raw scores; SUT forms stop mass q_l product_{j<l}(1-q_j). Both threshold cumulative mass, so avoid saying SUT only multiplies and never sums. Compare N+R versus expected depth, distinguish expected depth from executed steps/FLOPs. Explain more direct probabilistic interpretation without claiming universal superiority.

This roughly five-slide methods core leaves room within approximately ten slides for motivation, concrete CFQ task, results, and limitations/discussion. Keep flexibility after reading experiments. User is about to ask another reading question; invite it rather than starting deck production.

## State aggregation slide and structural comparison (2026-10-16)

User requests a separate slide immediately after basic dynamic halting: recursive interpolation versus expected halted representation. Distinguish this from UT recurrence through shared block application, which exists without halting. Added verified structural comparison master at docs/architecture_component_comparison.md covering MHA, generic MoE, vanilla Transformer, FFN-MoE Transformer, UT, MoA, MoH, SUT and MoM. MoA Figure 2 is a useful original projection/routing figure; MoM Figure 2 compares structural paths. Keep table about component inclusion, routing unit, reuse, and halting rather than benchmark performance. MoH/MoM are later 2024 comparisons, not prior sources for 2023 SUT.

## Final slide prepared (2026-10-16)

Prepared docs/slides/structural-comparison-final-slide.pptx as a standalone editable slide to append last when assembling the full deck. It removes component-only MHA and generic MoE rows. Columns: Architecture, MHA, FFN MoE, Router, Recurrence, Halting / skipping. Positive cells green, qualified Yes* light green, negatives white. Recurrence explicitly means reuse across depth. MoA/MoH MHA cells qualified as routed variants. UT halting optional; MoM reuse/skip qualified. Original detailed table remains a reference, not the requested slide layout.

Table 2 clarification: '- MoMHA' replaces MoMHA with MoA (Zhang et al. 2022), not MoE. MoA expert is one head, while MoMHA expert has H heads. MoA contains multiple experts/heads overall. FFN-MoE remains in this ablation. Reported SUT base 29.2 BLEU vs 28.7 for -MoMHA, both rounded 66M parameters. This is a replacement of attention formulations, not proof that H alone causes the difference without a matched detailed configuration audit.
