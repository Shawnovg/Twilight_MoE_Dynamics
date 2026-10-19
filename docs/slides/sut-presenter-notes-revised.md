# Sparse Universal Transformer: revised presenter notes

19 slides. Equations typeset from LaTeX; source retained in sut-equations.tex. Original experiment tables retained. Structural comparison is last.

## 1. Sparse Universal Transformer

Shawn Tan and Yikang Shen contributed equally. Affiliations reproduced from the title page. This presentation covers the paper and related architectural context.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 2. Universal Transformer: the shared backbone

Universal Transformers, Dehghani, Gouws, Vinyals, Uszkoreit and Kaiser, ICLR 2019. Original affiliations: University of Amsterdam, DeepMind and Google Brain. Encoder and decoder each reuse their own block over refinement depth. This is not encoder-decoder weight tying. Residual connections are not the source of recurrence. Positions update in parallel within a step, while depth steps depend on earlier steps. ACT is optional.

Sources:
Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3

## 3. Vanilla Transformer, UT and SUT

SUT Figure 1 is a schematic comparison. The bottom label has been corrected from Universal Sparse Transformer to Sparse Universal Transformer, matching the paper title. The original diagram is otherwise unchanged. Reuse reduces parameter duplication but does not eliminate repeated execution. A larger shared dense block is costly to execute at every depth. Sparse routing activates a subset of a larger expert pool.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 4. SUT block and sparse feedforward experts

SUT Figure 3 left. Uppercase K is used consistently in configuration tuples, corresponding to lowercase k in the paper. An FFN expert is a separate parameter set, reused across tokens and refinement steps. Router inputs are token representations, not rows with individually learned token-specific weights. Only selected experts are evaluated.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (ffn_tuple):
(E,K,D)

LaTeX (moe):
y=\sum_{e=1}^{E}g(e\mid h)\cdot f_e(h)

## 5. MoA: routing query and output projections

Original Figure 2 of Mixture of Attention Heads: Selecting Attention Heads Per Token, Zhang et al., 2022. In MoA, each expert is one attention head. Key and value projections are shared; each expert has its own query and output projections. The router chooses heads for each token and weights the corresponding outputs. SUT calls its multi-head-per-expert variant MoMHA. The figure is deliberately labeled as MoA, not as a diagram of SUT MoMHA.

Sources:
Zhang et al., Mixture of Attention Heads, 2022. Figure 2, Equations 6–12. https://arxiv.org/html/2210.05144
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (moa_att):
W_{i,t}^{\mathrm{att}}=\mathrm{Softmax}\!\left(\frac{q_t W_i^q(KW^k)^{\mathsf T}}{\sqrt{d_h}}\right)

LaTeX (moa_value):
o_{i,t}=W_{i,t}^{\mathrm{att}}VW^v

LaTeX (moa_output):
E_i(q_t,K,V)=o_{i,t}W_i^o

LaTeX (moa_sum):
y_t=\sum_{i\in G(q_t)}w_{i,t}\cdot E_i(q_t,K,V)

## 6. SUT’s mixture of multi-head attention

SUT Section 3.1 cites Zhang et al. 2022 (MoA) and Shen et al. 2023 (ModuleFormer). There is not a separately titled MoMHA paper among these references. ModuleFormer Figure 1a is a related sparse-attention/FFN block, not a detailed picture of SUT’s H-head grouping. ModuleFormer uses stick-breaking token attention, which must not be conflated with SUT’s stick-breaking halting over depth. SUT explicitly defines E experts of H heads with shared K/V projections and E query projection sets. W parameterizes a clipped relative-distance embedding range; it is not an attention visibility mask.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf
Zhang et al., Mixture of Attention Heads, 2022. Figure 2, Equations 6–12. https://arxiv.org/html/2210.05144
Shen et al., ModuleFormer: Learning Modular Large Language Models From Uncurated Data, 2023. Figure 1a and Section 3. https://arxiv.org/html/2306.04640v1

LaTeX (att_tuple):
(E,K,H,D,W)

## 7. Mutual-information maximization for routing

The main displayed expression is SUT Equation 1 in LaTeX, retaining the paper’s symbols. The loss symbol is calligraphic L, not capital Lambda. The marginal and entropy expansions clarify its terms without a numerical toy example. X is the collection of router inputs across token positions and refinement layers; g(e|h) is the router probability and g(e) its average. The printed expression is negative mutual information and should be minimized for the behavior described. The paper’s phrase to be maximised and the sign of its conditional-entropy underbrace are inconsistent with the displayed expression; we preserve the algebra and explain the sign consistently. The conditional entropy shown averages over router inputs.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (mim):
\mathcal{L}_{\mathrm{MIM}}=\sum_{e=1}^{E}g(e)\log g(e)-\frac{1}{|\mathcal{X}|}\sum_{h\in\mathcal{X}}\sum_{e=1}^{E}g(e\mid h)\log g(e\mid h)

LaTeX (marginal):
g(e)=\frac{1}{|\mathcal{X}|}\sum_{h\in\mathcal{X}}g(e\mid h)

LaTeX (entropy):
H(e)=-\sum_{e=1}^{E}g(e)\log g(e)

LaTeX (condentropy):
H(e\mid h)=-\frac{1}{|\mathcal{X}|}\sum_{h\in\mathcal{X}}\sum_{e=1}^{E}g(e\mid h)\log g(e\mid h)

LaTeX (mi_identity):
\mathcal{L}_{\mathrm{MIM}}=-H(e)+H(e\mid h)=-I(e;h)

## 8. Dynamic halting and ACT objectives

UT formulas summarize Appendix C and Graves ACT with a common index convention: t is token position, l is depth. q is the UT sigmoid-linear gate output, N the first threshold-crossing step, R the remaining mass assigned to that last update. R is one minus accumulated scores strictly before N, not the distance to the threshold. A maximum depth cap applies if the threshold is never reached. SUT Equations 2 and 4 retain their original symbols. Alpha-hat is a conditional halt probability from an MLP. Alpha is unconditional stopping mass, the conditional probability multiplied by survival through earlier layers. T counts token positions and L is the maximum refinement depth. The SUT threshold is .999. The UT penalty uses N+R, while SUT penalizes expected stopping depth. A finite-depth implementation must handle remaining mass. ACT penalties are not measured FLOPs.

Sources:
Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3
Graves, Adaptive Computation Time for Recurrent Neural Networks, 2016. https://arxiv.org/html/1603.08983
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (ut_stop):
N^{(t)}=\min\!\left\{n:\sum_{l=1}^{n}q_l^{(t)}>\tau\right\}

LaTeX (sut_stop):
\alpha_l^{(t)}=\hat\alpha_l^{(t)}\prod_{l'=1}^{l-1}\left(1-\hat\alpha_{l'}^{(t)}\right)

LaTeX (ut_remainder):
R^{(t)}=1-\sum_{l=1}^{N^{(t)}-1}q_l^{(t)}

LaTeX (sut_thresh):
\sum_{l'=1}^{l}\alpha_{l'}^{(t)}>\alpha_{\mathrm{thresh}},\qquad\alpha_{\mathrm{thresh}}=0.999

LaTeX (ut_act):
\mathcal{L}_{\mathrm{ACT}}^{\mathrm{UT}}=\frac{1}{T}\sum_{t=1}^{T}\left(N^{(t)}+R^{(t)}\right)

LaTeX (sut_act):
\mathcal{L}_{\mathrm{ACT}}=\frac{1}{T}\sum_{t=1}^{T}\sum_{l=1}^{L}\alpha_l^{(t)}\cdot l

## 9. UT: recurrence and recursive state interpolation

Original UT Figure 2 gives a compact view of the repeated encoder and decoder blocks. The recurrence reuses the same transformation parameters at every step and passes the current hidden states into the next step. Original UT Equation 4 is shown. Here superscript t follows UT and denotes refinement step, unlike t as token position in SUT. P^t, which enters the attention update, contains position and step signals. The lower interpolation equation is the formula quoted by SUT Section 3.2 when discussing prior UT halting. The alpha weight is an interpolation coefficient, not the SUT stopping mass. Each new state is mixed with the already mixed previous state, so later factors rescale earlier contributions. The final UT update uses the remainder. Recurrence exists even when dynamic halting is disabled.

Sources:
Dehghani et al., Universal Transformers, ICLR 2019. Figure 4 and Appendix C. https://arxiv.org/html/1807.03819v3
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (ut_update):
H^t=\mathrm{LayerNorm}\!\left(A^t+\mathrm{Transition}(A^t)\right)

LaTeX (recursive):
h_l=\alpha_l\cdot\hat h_l+(1-\alpha_l)\cdot h_{l-1}

## 10. SUT: expected halted states remain readable

SUT Equation 3 and Algorithm 1. h is the unhalted state, s is the expected halted state, and capital S stacks the s states across token positions. The first term is survival mass assigned to the current state; the second sums earlier stopping mass times the corresponding raw state. Active queries use their own previous unhalted h, while keys and values use S so they can read tokens that halted earlier. A halted token stops its own update but stays available to other tokens. The no-op branch copies h and s. Figure 3 right explains this cross-token interaction; it does not depict recurrence through sequence time.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX (sut_state):
s_l^{(t)}=\left(1-\sum_{l'=1}^{l-1}\alpha_{l'}^{(t)}\right)\cdot h_l^{(t)}+\sum_{l'=1}^{l-1}\alpha_{l'}^{(t)}h_{l'}^{(t)}

LaTeX (sut_query):
a_l^{(t)}=\mathrm{Attention}\!\left(\underbrace{h_{l-1}^{(t)}}_{Q},\underbrace{S_{l-1}}_{K},\underbrace{S_{l-1}}_{V}\right)

LaTeX (sut_noop):
h_l^{(t)}=h_{l-1}^{(t)},\qquad s_l^{(t)}=s_{l-1}^{(t)}

## 11. English–German translation: scale and cost

Input is English text, target is German text. Original Table 1 is embedded with all original rows, column headings and citation markers. MACs are the reported multiply–accumulate estimates, not wall-clock speed or FLOPs. LightConv and DynamicConv estimates are undercounted by the tool used, as the paper notes. Model configurations on the right come from the arXiv Appendix Table 6, absent from the supplied proceedings PDF. E is the number of experts in each pool; Datt is the table’s total attention-expert dimension, not the per-head D in the five-tuple. All four listed UT/SUT models use six encoder and six decoder steps. SUT big uses 110M parameters and 787M MACs versus dense UT big 105M and 3707M, at 29.4 versus 29.6 BLEU: approximately 4.7 times lower MACs, not a measured 4.7-times speedup. Relative to Transformer big, SUT big uses about half parameters and 38% of MACs. Training uses Adam, inverse-square-root learning-rate scheduling, label smoothing .1, 100 epochs and averaging the five best NLL checkpoints. Main paper Section 4.1 says vocabularies/embeddings are shared. Keep quality and cost trade-offs visible.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf
Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

## 12. Translation: ablations and FFN specialization

Table 2: removing MIM means replacing it with a load-balancing loss, not removing every routing regularizer. Minus MoMHA means replacement with MoA, not MoE. The reported replacement reduces BLEU by .5, the largest of these ablations, but does not isolate H alone while holding every design choice fixed. FFN experts remain. Halting removal costs .1 BLEU, ACT loss removal .2, MIM replacement .3. Table 3 explicitly concerns FFD/FFN expert–word co-occurrences across layers. The authors selected four experts showing patterns. Associations suggest specialization, but neither demonstrate causal functions nor quantify redundancy. Retain that distinction when connecting to our experiment.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 13. CFQ: familiar pieces, unfamiliar combinations

CFQ stands for Compositional Freebase Questions. The original Google Research animation illustrates how grammar building blocks compose into a sentence. The prediction task in SUT is semantic parsing: natural-language question to SPARQL query text. It is not predicting the parse tree shown by the animation, and the database answer is not the training target in this benchmark. MCD splits keep atomic distributions similar while changing combinations. The original animation is now shown as two still frames, avoiding viewer-dependent GIF playback. It illustrates grammar composition, not the neural prediction target. The animation’s source is linked below. SUT uses transformed query targets, so its preprocessing differs from an unmodified raw target comparison.

Sources:
Google Research, Measuring Compositional Generalization, 6 March 2020. Original animation: https://research.google/blog/measuring-compositional-generalization/
Keysers et al., Measuring Compositional Generalization, ICLR 2020. Figure 1. https://arxiv.org/pdf/1912.09713
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 14. CFQ results: the best setting uses one expert

All Table 4 models and means are shown. Original full uncertainty for our rows: T5-based UT MCD1 68.3±2.9, MCD2 43.1±1.5, MCD3 45.7±1.8, average52.3±1.6; UT without halting71.0±3.5,48.6±2.3,51.3±.2,average56.9±1.5; UT with halting72.4±3.5,51.1±1.8,51.7±2.3,average58.4±1.2. Those UT results average five seeds. Hyperparameters were selected on MCD1 validation. Attention (E=1,k=1,H=8,D=64,W=1), FFN(E=1,k=1,D=1024), embedding512, 8 encoder/8 decoder steps, 7.2M with halting and6.7M without. Thus the winning CFQ model is a UT special case, not evidence that multiple sparse experts are needed on CFQ. Target preprocessing and positional/norm implementation details improve baselines. Dangle is better at66.1 but pretrained and much more expensive in reported MACs. Avoid presenting all rows as a controlled architectural-only comparison.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf
Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

LaTeX (cfq_att):
(E,K,H,D,W)=(1,1,8,64,1)

LaTeX (cfq_ffn):
(E,K,D)=(1,1,1024)

## 15. Logical inference: relational reasoning beyond training

The task takes two propositional-logic expressions and predicts their semantic relation as a single label. Example a entails a OR b. The model receives sentence1 #SEP# sentence2, not a parse tree. Training has0–6 logical operators, evaluation7–12. The separate compositional splits A/B/C increase difficulty. The jump twice to jump jump example belongs to SCAN-style command interpretation, not this logical-inference task. Original SUT Figure2 illustrates a withheld combination of NOT and AND operations; it is not a literal input format for the model. The paper uses a12-layer encoder and12-layer decoder with attention(12,4,2,32,1), FFN(12,4,128), embedding512 and7.6M parameters. Halting is enabled. This is genuine multiple-expert SUT, unlike the CFQ winner.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf
Tan et al., arXiv version, Appendix A and Table 6. https://arxiv.org/pdf/2310.07096

LaTeX (logic_example):
a\quad\#\mathrm{SEP}\#\quad(a\lor b)

LaTeX (logic_att):
(E,K,H,D,W)=(12,4,2,32,1)

LaTeX (logic_ffn):
(E,K,D)=(12,4,128)

## 16. Logical inference: gains and remaining failures

Table5 values are accuracy percentages. SUT substantially exceeds the Transformer and LSTM on length extrapolation here. It also improves A and B, but C remains52 compared with LSTM59. Do not call compositional generalization solved. The source Figure4 shows average halting depth increasing with operator count. Operator count also correlates with length, so this does not isolate reasoning difficulty from sequence length. Baseline references: LSTM Bowman2015, Transformer Shen2019. The paper says a newly trained12-layer vanilla Transformer achieves approximately the same result as the cited baseline. The original figure caption says UT, while the surrounding setting and Table5 call the model SUT; describe it as the paper model without silently changing the source image.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 17. Post-training computation reduction

Section4.4 is valuable because it turns halting into a post-training control. Adjust the cumulative threshold and evaluate quality again. The original plots show fraction computed in blue and task quality in green. The estimates are based on decoder halting patterns with ground-truth source-target pairs, so these are skipped SUT block-step estimates, not measured end-to-end speedups. Logical inference has near-constant roughly50% skipped steps. CFQ text reports roughly33% skipped atthreshold.8 with stable quality in.8–.999, and about50% for a small .2% accuracy decrease; plots and prose have some inconsistencies, so avoid unsupported precision. Translation text reports9% less computation while retaining29.1 BLEU. Lowering threshold can alter token representations and subsequently other tokens’ halting, so monotonicity is not guaranteed. The cited images retain original axes.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 18. Conclusions and questions for our experiments

The structural hypothesis is that routing granularity, reuse across depth and stopping are separable choices. SUT combines them, but its experiments do not establish a universal principle. WMT supports parameter efficiency with a quality-cost trade-off; CFQ selects the one-expert UT setting; logical inference exposes failure on splitC. Token-expert associations are observational. For our controlled toy experiments, isolate changes, match parameter budgets and measure actual compute separately, select checkpoints on validation MSE, never expose regime labels to non-oracle models, and preserve negative results. These are proposed controls, not experimental findings. Toy-program results have not been added because they need a fresh audit and discussion; insert validated result slides before the final structural table later.

Sources:
Tan et al., Sparse Universal Transformer, EMNLP 2023. Sections 1–5, Tables 1–5. https://aclanthology.org/2023.emnlp-main.12.pdf

## 19. Structural comparison

Final comparison slide for the planned SUT presentation. MHA means multi-head attention, including routed variants; it does not imply a full MHA block inside every MoA expert. MoA routes single heads, SUT MoMHA has H heads per expert. FFN MoE means a separately routed FFN expert pool. Router lists the selected unit, blank means no expert/head router (UT still has a halting gate). Recurrence means reuse of parameters across depth, not recursive state interpolation. Typical FFN-MoE shown uses separate layer pools. UT halting is optional. MoM reuses a module pool rather than one fixed block, and SKIP is optional; skipping does not imply permanent halting. MoM attention assembles projection parameters. MoH shared heads are always active, not necessarily shared KV projections. MoH and MoM are later 2024 comparisons, not predecessors of SUT.
Sources:
Transformer: https://arxiv.org/html/1706.03762
Switch: https://arxiv.org/html/2101.03961
UT: https://arxiv.org/html/1807.03819v3
MoA Sections 4-5: https://arxiv.org/html/2210.05144
MoH Section 3.2: https://arxiv.org/html/2410.11842
SUT Sections 3.1-3.2: https://aclanthology.org/2023.emnlp-main.12.pdf
MoM Sections 3.1-3.3: https://arxiv.org/html/2407.06677
