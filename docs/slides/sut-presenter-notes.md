# Sparse Universal Transformer: presenter notes

17 slides. The approved structural comparison is last. Toy-program results are not included.

## 1. Sparse Universal Transformer

The paper asks whether a recurrent Transformer can retain the benefits of shared parameters while scaling its capacity economically. We will separate the shared-depth backbone, sparse selection of computation, and learned stopping. The final table will compare these structural choices across related architectures. This deck covers the paper, not results from our toy experiment.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 2. Why sparsify a Universal Transformer?

A vanilla Transformer allocates different parameters to successive layers. A UT reuses a block. At the same width, reuse reduces parameters but does not remove repeated computation. At matched total parameter count, a dense UT may require a larger repeated block. SUT activates only selected attention and FFN experts at each step. The original Figure 1 is a schematic, not a FLOP or runtime measurement. Avoid interpreting its illustrative scaling factors as universal laws.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 3. Universal Transformer: the shared backbone

Use the original UT Figure 4 as the architectural anchor. Encoder and decoder each reuse their own block across refinement steps; this does not mean the encoder and decoder have identical parameters. Token positions update in parallel within a step, while refinement steps depend on earlier steps. Residual connections are not the source of recurrence. ACT is optional. This full diagram includes position and step signals, dropout, normalization and decoder cross-attention. We do not need to derive their details in this presentation.

Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3

## 4. SUT block and sparse feedforward experts

Figure 3 left locates the changes on the UT scaffold. SUT routes attention and FFN computation separately and reuses this block across depth. For a token state h, the router picks top-k experts and weights their outputs. The E experts are different parameter sets, reused across tokens and refinement steps. A token does not receive its own independently learned parameter matrix. The attention unit is a multi-head expert, which is detailed next. Auxiliary losses train these choices but are not extra inference blocks.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 5. Attention experts: MoA and SUT’s MoMHA

Original MoA Figure 2 shows shared K/V projections, expert-specific query and output projections, and router-weighted output aggregation. These are paired projection banks with the same expert indices, not an extra FFN mixture. In MoA each expert is one head. SUT MoMHA uses H heads per expert: E sets of query projections and one H-head set of K/V projections shared across experts. The displayed equations describe the MoA head calculation; SUT extends the expert to multiple heads and uses relative positions. W is the relative-position range, not a hard restriction that attention can only see nearby tokens.

Zhang et al., Mixture of Attention Heads, 2022. Figure 2, Equations 6–12. https://arxiv.org/html/2210.05144

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 6. Mutual-information maximization for routing

Minimize the printed negative mutual-information expression: minus marginal entropy plus conditional entropy. High marginal entropy distributes routing probability across the expert pool. Low conditional entropy encourages confident input-specific choices. This does not guarantee semantic specialization or perfectly equal discrete top-k loads. The toy numbers use natural logs and two experts. The paper prose says maximized beside the negative-MI equation; present the mathematically consistent minimization interpretation. Our reading audit also found that the current public repository appears to combine entropy signs differently. This slide explains the paper objective, not a verified reproduction of its implementation; resolve that discrepancy before implementing a comparison.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 7. Dynamic halting: addition versus stick breaking

Use common notation: l is refinement depth and q_l is the gate output for one token. UT adds raw gate scores until their sum crosses a threshold; at the last step it uses a remainder. SUT interprets q_l conditionally on surviving earlier decisions, so alpha_l equals q_l times the preceding survival product. Both use a cumulative stopping criterion. The example fixes q=.5 and threshold=.9 to isolate arithmetic; actual SUT threshold is .999. It does not show that trained SUT is inherently slower. UT ACT uses N+R; SUT uses expected depth. At a finite cap, remaining mass needs a terminal assignment. Expected depth is not executed depth or FLOPs.

Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3

Graves, Adaptive Computation Time for Recurrent Neural Networks, 2016. https://arxiv.org/html/1603.08983

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 8. Combining states across refinement steps

Separate recurrence of the shared block from recursion in the output combination. UT Appendix C recursively interpolates a transformed state v_l and the previous accumulated output u_(l-1); repeated later gates change earlier contributions. SUT explicitly weights prior raw states by stopping probabilities, with survival mass on the current raw state. Its active token uses h_(l-1) as query input and S_(l-1) as K/V input, where capital S stacks halt-aware states for all tokens. Other token positions may already have stopped updating but their information remains readable. No-op means preserve state and skip its own update. Decoder masks still apply. Original Graves ACT has a direct weighted sum, so do not conflate it with UT Appendix C.

Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 9. English–German translation: scale and cost

Input is English text, target is German text. Table 1 is reproduced completely, with abbreviated labels. MACs are the reported multiply–accumulate estimates, not wall-clock speed or FLOPs. LightConv and DynamicConv estimates are undercounted by the tool used, as the paper notes. Model configurations on the right come from the arXiv Appendix Table 6, absent from the supplied proceedings PDF. E is the number of experts in each pool; Datt is the table’s total attention-expert dimension, not the per-head D in the five-tuple. All four listed UT/SUT models use six encoder and six decoder steps. SUT big uses 110M parameters and 787M MACs versus dense UT big 105M and 3707M, at 29.4 versus 29.6 BLEU: approximately 4.7 times lower MACs, not a measured 4.7-times speedup. Relative to Transformer big, SUT big uses about half parameters and 38% of MACs. Training uses Adam, inverse-square-root learning-rate scheduling, label smoothing .1, 100 epochs and averaging the five best NLL checkpoints. Main paper Section 4.1 says vocabularies/embeddings are shared. Keep quality and cost trade-offs visible.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

## 10. Translation: ablations and FFN specialization

Table 2: removing MIM means replacing it with a load-balancing loss, not removing every routing regularizer. Minus MoMHA means replacement with MoA, not MoE. The reported replacement reduces BLEU by .5, the largest of these ablations, but does not isolate H alone while holding every design choice fixed. FFN experts remain. Halting removal costs .1 BLEU, ACT loss removal .2, MIM replacement .3. Table 3 explicitly concerns FFD/FFN expert–word co-occurrences across layers. The authors selected four experts showing patterns. Associations suggest specialization, but neither demonstrate causal functions nor quantify redundancy. Retain that distinction when connecting to our experiment.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 11. CFQ: familiar pieces, unfamiliar combinations

CFQ stands for Compositional Freebase Questions. The original Google Research animation illustrates how grammar building blocks compose into a sentence. The prediction task in SUT is semantic parsing: natural-language question to SPARQL query text. It is not predicting the parse tree shown by the animation, and the database answer is not the training target in this benchmark. MCD splits keep atomic distributions similar while changing combinations. The original CFQ Figure 1 at the bottom describes dataset generation, with an intermediate logical form. This logical form is not an extra required neural output. The animation is embedded as GIF; if the presentation viewer shows a still image, open the linked original article. SUT uses transformed query targets, so its preprocessing differs from an unmodified raw target comparison.

Google Research, Measuring Compositional Generalization, 6 March 2020. Original animation: https://research.google/blog/measuring-compositional-generalization/

Keysers et al., Measuring Compositional Generalization, ICLR 2020. Figure 1. https://arxiv.org/pdf/1912.09713

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 12. CFQ results: the best setting uses one expert

All Table 4 models and means are shown. Original full uncertainty for our rows: T5-based UT MCD1 68.3±2.9, MCD2 43.1±1.5, MCD3 45.7±1.8, average52.3±1.6; UT without halting71.0±3.5,48.6±2.3,51.3±.2,average56.9±1.5; UT with halting72.4±3.5,51.1±1.8,51.7±2.3,average58.4±1.2. Those UT results average five seeds. Hyperparameters were selected on MCD1 validation. Attention (E=1,k=1,H=8,D=64,W=1), FFN(E=1,k=1,D=1024), embedding512, 8 encoder/8 decoder steps, 7.2M with halting and6.7M without. Thus the winning CFQ model is a UT special case, not evidence that multiple sparse experts are needed on CFQ. Target preprocessing and positional/norm implementation details improve baselines. Dangle is better at66.1 but pretrained and much more expensive in reported MACs. Avoid presenting all rows as a controlled architectural-only comparison.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

## 13. Logical inference: relational reasoning beyond training

The task takes two propositional-logic expressions and predicts their semantic relation as a single label. Example a entails a OR b. The model receives sentence1 #SEP# sentence2, not a parse tree. Training has0–6 logical operators, evaluation7–12. The separate compositional splits A/B/C increase difficulty. Original SUT Figure2 illustrates a withheld combination of operations; it is not a literal input format for the model. The paper uses a12-layer encoder and12-layer decoder with attention(12,4,2,32,1), FFN(12,4,128), embedding512 and7.6M parameters. Halting is enabled. This is genuine multiple-expert SUT, unlike the CFQ winner.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

## 14. Logical inference: gains and remaining failures

Table5 values are accuracy percentages. SUT substantially exceeds the Transformer and LSTM on length extrapolation here. It also improves A and B, but C remains52 compared with LSTM59. Do not call compositional generalization solved. The source Figure4 shows average halting depth increasing with operator count. Operator count also correlates with length, so this does not isolate reasoning difficulty from sequence length. Baseline references: LSTM Bowman2015, Transformer Shen2019. The paper says a newly trained12-layer vanilla Transformer achieves approximately the same result as the cited baseline. The original figure caption says UT, while the surrounding setting and Table5 call the model SUT; describe it as the paper model without silently changing the source image.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 15. Post-training computation reduction

Section4.4 is valuable because it turns halting into a post-training control. Adjust the cumulative threshold and evaluate quality again. The original plots show fraction computed in blue and task quality in green. The estimates are based on decoder halting patterns with ground-truth source-target pairs, so these are skipped SUT block-step estimates, not measured end-to-end speedups. Logical inference has near-constant roughly50% skipped steps. CFQ text reports roughly33% skipped atthreshold.8 with stable quality in.8–.999, and about50% for a small .2% accuracy decrease; plots and prose have some inconsistencies, so avoid unsupported precision. Translation text reports9% less computation while retaining29.1 BLEU. Lowering threshold can alter token representations and subsequently other tokens’ halting, so monotonicity is not guaranteed. The cited images retain original axes.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 16. Conclusions and questions for our experiments

The structural hypothesis is that routing granularity, reuse across depth and stopping are separable choices. SUT combines them, but its experiments do not establish a universal principle. WMT supports parameter efficiency with a quality-cost trade-off; CFQ selects the one-expert UT setting; logical inference exposes failure on splitC. Token-expert associations are observational. For our controlled toy experiments, isolate changes, match parameter budgets and measure actual compute separately, select checkpoints on validation MSE, never expose regime labels to non-oracle models, and preserve negative results. These are proposed controls, not experimental findings. Toy-program results have not been added because they need a fresh audit and discussion; insert validated result slides before the final structural table later.

Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 17. Structural comparison

Final comparison slide for the planned SUT presentation. MHA means multi-head attention, including routed variants; it does not imply a full MHA block inside every MoA expert. MoA routes single heads, SUT MoMHA has H heads per expert. FFN MoE means a separately routed FFN expert pool. Router lists the selected unit, blank means no expert/head router (UT still has a halting gate). Recurrence means reuse of parameters across depth, not recursive state interpolation. Typical FFN-MoE shown uses separate layer pools. UT halting is optional. MoM reuses a module pool rather than one fixed block, and SKIP is optional; skipping does not imply permanent halting. MoM attention assembles projection parameters. MoH shared heads are always active, not necessarily shared KV projections. MoH and MoM are later 2024 comparisons, not predecessors of SUT.
Sources:
Transformer: https://arxiv.org/html/1706.03762
Switch: https://arxiv.org/html/2101.03961
UT: https://arxiv.org/html/1807.03819v3
MoA Sections 4-5: https://arxiv.org/html/2210.05144
MoH Section 3.2: https://arxiv.org/html/2410.11842
SUT Sections 3.1-3.2: https://aclanthology.org/2023.emnlp-main.12.pdf
MoM Sections 3.1-3.3: https://arxiv.org/html/2407.06677


