# CFQ task and composition schematic

Status: superseded for the planned slides. User prefers original reference-paper figures (CFQ Figure 1 and Appendix A) and found the dual use of A/B confusing. Do not use this custom schematic by default.

Original explanatory schematic for the reading discussion and later slides. PNG and editable vector SVG share the same basename.

All example questions and the split are illustrative, not reproduced dataset records. Query relation names are shortened; the colon-prefixed names require a namespace declaration to execute. The figure explains question-to-query prediction, not an executable benchmark item.

The top panel separates the prediction target (query text) from the answer returned by database execution. The lower panel illustrates recombination of familiar relations. Actual CFQ splits compare distributions of generation-rule atoms and compounds; they are not defined by this three-example toy training set.

Sources:
- Keysers et al., Measuring Compositional Generalization: A Comprehensive Method on Realistic Data, ICLR 2020: https://arxiv.org/abs/1912.09713
- Authors' explanation: https://www.research.google/blog/measuring-compositional-generalization/
- Tan et al., Sparse Universal Transformer, EMNLP 2023, Section 4.2: https://aclanthology.org/2023.emnlp-main.12/

Suggested slide narration: The model learns a translation from a natural-language question into a structured query. Compositional evaluation asks whether it can construct the correct query when familiar semantic pieces appear in unfamiliar combinations.
