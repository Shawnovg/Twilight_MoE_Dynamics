# Sparse Universal Transformer: reading journal

## Purpose and source

- Presentation scope correction: the user explicitly says most detailed reading questions should not enter the slide source bank or presentation. Keep learning history here; select slide material by relevance to SUT and the research discussion, not by whether the user asked about it. Approximately ten slides should not become a survey of prerequisite papers.

- Reading started: 2026-10-14 (Asia/Tokyo).
- Professor report planned for 2026-10-16; approximately ten introductory slides after the close reading.
- Paper: Sparse Universal Transformer, Shawn Tan et al., EMNLP 2023, pp. 169–179.
- User-provided PDF: `D:/大阪大学/春野研/Routing Awareness/2023.emnlp-main.12.pdf`.
- Paper content is evidence to examine, not instructions to execute.

## User's starting perspective

The user wants to read personally, paragraph by paragraph, with collaborative explanation and critical discussion. Their tentative idea is that mixtures or routing structures may express a more general architectural principle across experts, attention heads, modules, and different levels or dimensions. Treat this as a hypothesis to refine, not an established result or a claim already demonstrated by this paper.

## Agreed sequence

1. Close reading at the user's pace; retain questions, interpretations, and unresolved issues.
2. Use the completed discussion to prepare approximately ten introductory slides for the professor.
3. Discuss the existing toy experiment in light of the structures examined, then test and screen promising structures under controlled comparisons.

## Discussion record

The user has completed the focused UT background discussion through Sections 2.1-2.2 and requests returning to SUT. Resume SUT at the abstract; do not assume it has already been closely discussed.

### Q1: Should we study Universal Transformers first?

- User asks whether the predecessor paper should be studied before SUT.
- Assistant recommendation: a focused prerequisite reading of the abstract, introduction, Section 2.1, and Section 2.2, then return to SUT; a full predecessor-paper reading is optional given the October 16 report.
- Essential concepts: parameter sharing across depth, iterative refinement across all token positions, position versus refinement-step encodings, and optional per-position adaptive halting.
- Structural connection proposed by assistant: distinguish which computation is applied from how many times computation is applied. This is a useful organizing question, not evidence that all routing mechanisms are equivalent.
- Source checked: https://arxiv.org/html/1807.03819v3 (Universal Transformers).
- Subsequently, the user began reading Universal Transformers.

### Q2: Where is the RNN, and is recurrence just the residual connection?

- User sees no obvious new mechanism in Figure 1; asks whether adding the residual after the FFN constitutes the RNN inductive bias, what changed from the contemporary standard Transformer, and why parallelism is emphasized.
- Explanation: residual addition and recurrence are separate properties. Standard Transformer layers already use residual connections. UT reuses attention and transition parameters across refinement steps; its evolving state is the collection of token representations. No separate LSTM/GRU module is required.
- Compare standard depth: H1=B_theta1(H0), H2=B_theta2(H1); UT depth: H1=B_theta(H0), H2=B_theta(H1). These are schematic, omitting position/step encodings and internal operations.
- Parallelism concerns token positions within one step, not dependent refinement steps. Standard Transformer also parallelizes positions; UT preserves that benefit while adding depth recurrence. Autoregressive output generation remains sequential across generated tokens.
- Inductive bias: sharing encourages reuse of an update rule across steps; it does not guarantee successful algorithm learning or extrapolation. An unrolled fixed-step UT is a feedforward computation graph with tied parameters.
- Dates: original Transformer 2017; UT preprint July 2018, ICLR 2019 publication.
- Slide candidate: side-by-side distinct-parameter stack versus shared-parameter unrolled loop; explicitly distinguish residuals, sharing, and adaptive depth.
- Sources: https://arxiv.org/html/1807.03819v3 and https://arxiv.org/abs/1706.03762.

For each passage, record the page/section, the user's actual question, explanation, evidence limits, unresolved questions, and any slide or experiment implications. Keep user ideas distinct from assistant suggestions and author claims. Revise interpretations when later evidence changes them.

### Q3: Intuition for P^t and equations for ACT

- User has nearly completed UT Sections 2.1 and 2.2. Understands reuse of encoder/decoder blocks; asks why P^t is added and requests mathematical explanation and simple examples of Appendix C.
- P_i^t = PE(i) + PE(t): fixed sinusoidal position and refinement-step signals, giving the shared computation explicit coordinates. Step-dependent inputs do not imply step-dependent learned weights. Encoder and decoder each reuse their own block; no implication that their weights are identical.
- ACT explanation: learned scalar sigmoid score, cumulative mass, threshold crossing, last-step remainder, and per-position stopping. Use threshold 0.99 with illustrative scores 0.2, 0.3, 0.6: final update weight 0.5, total mass 1.
- Important close-reading finding: UT Appendix C Listing 2 uses z_t = a_t*u_t + (1-a_t)*z_(t-1), with z_0=0, rather than original Graves ACT's sum_t a_t*u_t. For illustrative scalar states 2,4,8 and weights 0.2,0.3,0.5, appendix yields 4.74 whereas original ACT weighted sum yields 5.6. Do not silently equate these rules.
- Appendix returns transformed_state as next raw state, and new_state separately as the blended output. Thus output freezing does not establish that all raw-state computation is skipped. Prior explanation of halting was conceptual; distinguish literal listing from prose and measure runtime separately in future experiments.
- Original ACT includes task loss plus ponder penalty proportional to N+R; hard threshold changes are not globally smooth. Appendix C alone does not specify that training objective.
- Sources checked: https://arxiv.org/html/1807.03819v3 (Equations 4-7, Section 2.2, Appendix C) and https://arxiv.org/html/1603.08983 (Equations 5-13).
- Assistant-proposed structural connection: explicit depth conditioning and adaptive depth are distinct from expert selection; retain as comparison axes, not a demonstrated unification.

### Q4: How do sinusoids encode position, and what is the ACT state?

- User asks whether position i is frequency, how i / 10000^(2j/d) should be parsed, what the curves look like, why alternating sine/cosine dimensions are used, and whether s_t is one row of H^t.
- Clarify: i is position, j indexes dimension pairs, omega_j=10000^(-2j/d) is angular frequency; 2j/d is an exponent on 10000. Even/odd dimension indices 2j and 2j+1 pair sine and cosine at the same frequency; this does not branch on the parity of j or token position.
- Interpretation: many clock hands at different speeds; position is represented by their joint phases. Pair gives both circle coordinates, though each pair remains periodic. Relative shifts are rotations; pair dot product equals cos(omega_j*(i-k)). Multiple frequencies encode several scales without guaranteeing unlimited distinguishability or extrapolation.
- Created mathematical plot for d=8, pairs j=0,1,2, i=0..70. Angular frequencies 1, 0.1, 0.01 radians/position. Image: C:/Users/ASUS/.codex/visualizations/2026/10/14/01a1395d-1945-7603-b36b-4a6a11352a74/position-encoding.png.
- State notation clarification: earlier s_t was incoming state for step t, so with token index restored s_(i,t) corresponds schematically to row i of H^(t-1); H_i^t is the updated row. Appendix C raw state and blended previous_state/new_state are distinct, so exact correspondence requires care.

## Assistant-proposed reading lens

### Q5: Regular languages, compositional generalization, and CFQ (2026-10-15)

- User finished SUT abstract and introduction. Asks whether regular-language difficulty means generally poor language performance, what compositional generalization and Figure 2 mean, and what CFQ contains.
- Explain regular as a technical formal-language class recognizable by finite-state automata, not ordinary natural language. Illustrative even-number-of-1s language has two states. Separate successful fitting/in-distribution testing from extrapolation to longer sequences. Delétang et al. report task-dependent failures, not failure on every regular task (Even Pairs succeeds).
- Explain compositional generalization as applying known components/rules in held-out combinations. Deliberate structural splits differ from ordinary random splits and are distinct from length-only generalization.
- Visually inspected full SUT PDF page 2: Figure 2 depicts not and and at the first operation stage during training, but and above not at test time. It is a schematic logical-operation example credited to Shen et al., not a CFQ example. Do not interpret partial drawing as a complete binary Boolean formula.
- CFQ expands to Compositional Freebase Questions. Generated natural-language questions paired with structured queries; in SUT the task is question-to-SPARQL semantic parsing. Freebase is the knowledge base. Illustrative conjunction of directed/wrote constraints tests recombination; examples we invent must be labeled as such. Atom distributions stay similar while compound distributions differ; atoms are generation rules, not merely words. MCD = maximum compound divergence; MCD1/2/3 are split instances, not difficulty levels.
- SUT evaluates compositional generalization using CFQ and a separate logical-inference task. Section 4.2 selected E=k=1, the UT setting, so CFQ evidence should not later be misreported as proof of multi-expert specialization.
- Sources: local SUT PDF pp. 169-170, 173-174; https://arxiv.org/html/2207.02098v2 Section 5.4; https://www.research.google/blog/measuring-compositional-generalization/.

### Transition back to SUT

- User accepts the encoding intuition as an information-bearing code and does not need more sinusoidal detail now.
- Brief clarification to retain: sinusoidal codes are fixed; the model learns how to use them. Vector dimensions provide channels for the code, rather than additional position metadata that must itself be decoded.
- UT prerequisites covered: shared blocks across depth, token-position parallelism versus sequential refinement, position/step encoding, and adaptive halting.
- Next reading focus proposed: SUT abstract, distinguishing inherited UT mechanisms from sparse expert computation and the changed halting mechanism. User retains control of reading pace.

Track what is shared, what is selected or mixed, at what granularity selection occurs, and how computational depth is allocated. Examine whether analogous structures have the same function rather than assuming that similar diagrams imply equivalent mechanisms.

## Experiment constraints

### Q10: Main architecture figure for slides

- User is starting SUT Section 3 and explicitly requires original UT Figure 4 to have an important position in the slides. Explain SUT amendments by mapping them onto that familiar detailed structure.
- Confirmed SUT has its own Figure 3 on PDF page 4 (proceedings p.172); inspected complete rendered page. Left: routers with attention and MLP experts, layer normalization and residual pathways; right: repeated computation and halting-state flow.
- Planned comparison: UT Figure 4 as baseline, SUT Figure 3 as companion; show feed-forward experts, MoMHA, and changed halting/attention-state flow. User did not request creating the slide deck yet; preserve pace of close reading.

### Q11: Dropout versus RouterInterp top-k

- User associates dropout with previously read RouterInterp feature selection. Confirmed RouterInterp exists and uses Top-K/BatchTopK SAEs as well as ranking features predictive of expert routing; do not claim to remember a prior reading absent from this chat.
- Clarify random training-time activation masking versus score-ranked selection. Top-k SAE latents, top-n explanatory features, and top-k MoE experts are distinct uses of sparsity. Dropout does not select the strongest features and is ordinarily disabled at inference.
- Source checked: https://arxiv.org/html/2610.11775v1 . Keep this clarification in reading notes, not automatically in slides.

### Q12: Brief orientation to MIM, stick-breaking halting, and ACT

- User asks for short conceptual explanations before continuing Method, not full derivations.
- MIM intended behavior: sharp expert choices per input but diverse usage across inputs, avoiding one-expert collapse and uniformly indecisive routing. Standard mutual information I(H;E)=H(E)-H(E|H); maximize it or minimize its negative. SUT printed Eq.1 has negative-MI sign while its prose says maximized: flag for detailed equation reading later rather than silently endorsing the sign.
- Stick-breaking: predictor gives conditional halt probability q_l; actual halt mass alpha_l=q_l product_(j<l)(1-q_j). Original UT ACT instead adds sigmoid scores until threshold, then fills remainder. Both may threshold cumulative mass; the mass construction differs. SUT uses an MLP halt predictor, not necessarily just the original sigmoid affine projection. Notation P^t in UT is positional/step encoding and unrelated to halt probability p_t.
- ACT means adaptive computation time: variable refinement steps per token. SUT's ACT auxiliary objective is the mean expected halting depth, sum_l l*alpha_l, weighted against task loss. It encourages earlier stopping; UT already had adaptive halting, so adaptive depth itself is not new here.

### Q13: MoMHA structure and W

- User explicitly prioritizes Section 3.1 sparse mixture definition, three-tuple FFN configuration and five-tuple MoMHA configuration for slides. They connect it to prior attention-expert notes, emphasizing shared K/V projections, expert-specific Q projections, weighted output aggregation, residual/norm and FFN flow.
- Correct gently: a Q row is a token query vector, not a token-specific trainable matrix. Query projection parameters are expert/head-specific and reused across tokens. Router g(e|h) scores experts from input h; do not assume it consumes projected Q. H is the number of heads per expert; D is per-head dimension.
- W explanation: W relative offsets before, W after, and current position, totaling 2W+1 learned relative-position embeddings. Example W=2 gives -2,-1,0,+1,+2. These describe relative token positions, not depth or token identities. Does not alone imply local-only attention; behavior beyond window needs implementation verification.

### Q14: Is W a sliding window, and why absent from MoA?

- Clarify window as a relative-coordinate range centered on each query token, not a sequential scan or necessarily local attention masking. Positions can be processed in parallel.
- Purpose: supply direction/distance information for attention; content-based expert selection does not itself specify word order. Relative relations can be reused at different absolute locations. SUT introduction cites prior evidence for relative positional embeddings in UT compositional tasks; this is motivation, not proof that a particular W is optimal.
- Checked original MoA https://arxiv.org/html/2210.05144 : its core routing/attention formulation does not introduce this W parameter. Do not claim MoA lacks all positional information. Distinguish expert selection (k) from relative-position configuration (W).

### Q15: Why bound relative offsets instead of encoding all positions?

- User clarifies they understand positional encoding; asks specifically why introduce a bounded window. Explain distinction between per-token absolute coordinates and per-pair relative offsets. An unbounded learned distance lookup would need entries for new distances; finite W fixes table size and allows fine near-distance encoding with coarser far-distance treatment. This is a design trade-off, not a necessity; sinusoidal or other schemes can encode all positions without such a learned lookup window.
- Attempted author-code verification: repository https://github.com/shawntan/SUT located, including sut_layer/moa_attention.py, but file retrieval failed (web cache miss and shell DNS failure). Do not assert SUT clips offsets as a verified fact. If illustrating clipping, explicitly label it a common implementation, not confirmed here.
- Correct earlier moving-window imagery: this is an embedding vocabulary/range, not a container that removes distant tokens or an inherently sequential sliding computation. General rationale is architectural interpretation, not a demonstrated SUT ablation of W.

### Q16: Applying relative embeddings while processing the whole sequence

- User's core confusion: assumes W encoding is preprocessing added once to each token embedding, and cannot see how a short window handles the full sequence.
- Successfully read author source with approved network access: https://github.com/shawntan/SUT/blob/main/sut_layer/moa_attention.py . rel_pos_logits forms the full pairwise index grid j-i, clips it, computes query-relative embedding dot products, gathers by pairwise offsets, and adds these to QK logits before attention softmax. Thus the operation is inside attention, not merely a short vector added to the input sequence.
- Explain with N=5 and illustrative W=2: full 5x5 clipped offset matrix; each cell chooses one of five shared relative vectors. Scores are q_i dot k_j plus q_i dot r_clip(j-i), omitting scaling/head/expert/batch indices. Full sequence remains present. Each batch member uses the same relative lookup.
- Exact public code detail for future implementation: max_positions=64, table shape [num_heads,head_dim,129], actual clipping bounds -63..63 plus offset64 (outermost slots unused). Avoid presenting the illustrative W2 clipping matrix as literal public-code configuration. Public repo describes WMT implementation; do not extrapolate its constant to all paper experiments.
- Prior inability to verify clipping is now resolved for this public implementation. Record distinction between paper's parameterization and implementation boundary convention.

### Q17: How Equation 1 MIM works

- User now requests detailed understanding of Equation 1 and how its auxiliary objective achieves routing goals.
- Explain g(e|h) conditional routing, g(e) mean across router inputs, entropy as uncertainty, and negative MI = -H(E)+H(E|H). Minimizing printed equation encourages balanced average probability mass and sharp per-input decisions. Maximize MI equivalently; printed prose saying maximize the negative-MI equation is inconsistent.
- Illustrative two-expert cases with natural logs: collapse (all [1,0]) H(E)=0,H(E|H)=0,L=0; uniform every input ([.5,.5]) .693,.693,0; evenly split confident routing .693,0,-.693. Balanced probability mass is not a guarantee of equal discrete top-k counts or semantically meaningful expert specialization.
- IMPORTANT implementation audit: fetched public sut_layer/parallel_linear/parallel_experts/moe.py and sut_loss.py. Code sets H_e=sum p_e log p_e (negative marginal entropy); neg_H_e_given_x=sum p_x*p*log p (negative conditional entropy), then miloss=-(neg_H_e_given_x+H_e), i.e. sum of entropies for nonnegative normalized p_x. This does not resolve paper sign mismatch and is not standard negative MI. Returned aux losses are added to task loss. Do not claim the repository validates the intended equation or infer exact published training behavior from current public code. Need further version/config audit if implementation becomes a task; user currently asks conceptual equation explanation.

### Q6 and requested teaching figures (2026-10-15)

- User asked explicitly what CFQ model inputs and targets are; explanation used an invented English question mapped to query text, separating query prediction from database execution and returned names. User now understands.
- User requests conceptual task/dataset illustrations for this series of studies and the later slides. Prepared docs/figures/cfq-task-and-composition.png and editable .svg, with source/caption notes in the matching .md. Clearly labeled as invented examples, simplified query names, and a toy split rather than actual CFQ items/MCD split.
- Preserve a preference for concrete input-target examples and structural figures in the final presentation. Add relevant diagrams as later tasks and mechanisms arise.
- Reading progress: user is proceeding to SUT Section 2, Background and Related Work.

### Figure preference correction

- User found the invented Film A/Film B placeholders and panel A/B labels confusing, and explicitly prefers existing figures from reference papers over newly drawn schematics.
- Prefer original cited figures in future explanations and slides. The custom CFQ schematic is superseded as the planned presentation visual; preserve it but do not use by default.
- Located original CFQ paper Figure 1 (question, intermediate logical form, SPARQL query, and generation rules) and Appendix A (actual dataset record). Figure 1 describes dataset construction, not a required neural-model pipeline.
- Source: https://arxiv.org/html/1912.09713v2#S3.F1 ; Appendix A: https://arxiv.org/html/1912.09713v2#A1 .

### Q7: CFQ target, computational limits, tied weights, and the ALBERT contrast

- User asks whether CFQ Figure 1 requires predicting logical form or SPARQL. Answer: SPARQL query text in this benchmark; logical form is an intermediate in dataset generation, not an additional prediction target.
- User requests more original-reference materials for slides. Created docs/sut_slide_source_bank.md with original figure/table candidates and scoped theoretical findings.
- User asks what tied means: identical shared trainable parameter objects across depth, not frozen parameters and not dense/sparse. Gradients from repeated uses update the same shared weights.
- User asks explicit theoretical limitations: summarize hard-attention parity/nesting limitations, distinguish Hahn's soft-attention sensitivity/loss results, and Merrill's TC0 result. Qualify all claims by attention, precision, and fixed-depth assumptions. Liu also proves positive shortcut constructions; no blanket claim that all Transformers cannot reason or recognize regular languages.
- Exact referents in SUT Section 2: authors reporting the dip are ALBERT authors (Lan et al.); contrasting prior work is original UT (Dehghani et al. 2018). Dip means reduced task performance when sharing at the compared configuration, not a transient training dip. ALBERT Table 4 E=128 average 81.6 unshared vs 80.1 all-shared, with 89M vs 12M parameters. Distinguish this from scaling wider ALBERT variants and from different UT tasks/training protocols.

### Q8: Takase and Kiyono sharing strategies

- User requests a brief explanation of the 2021 reference's strategies. Explain partial sharing with three parameter sets across six layers: sequence (1,1,2,2,3,3), cycle (1,2,3,1,2,3), cycle-rev (1,2,3,3,2,1). General cycle-rev reverses the final cycle only.
- These are fixed assignments across depth, not token-conditioned routing. Aim is parameter/computation trade-off using multiple smaller shared blocks. Added original Figure 1 to slide source bank.

### Q9: Why would multiple learned blocks reduce computation?

- User correctly challenges the cost claim: computation depends on executed depth and width, and extra distinct blocks increase trainable parameters at unchanged width.
- Clarify two comparisons: fixed depth and block dimensions gives roughly equal forward/backward arithmetic regardless of sharing, with more parameter/optimizer storage for partial sharing; matched total parameter budget allows smaller blocks in partial sharing than in a single-wide-block UT.
- Schematic block-parameter budget B, executed steps N, distinct blocks M: fully shared uses N evaluations of a B-parameter block; partial sharing uses N evaluations of blocks with roughly B/M parameters each. NP-style arithmetic is only a rough projection/FFN proxy; attention sequence-length costs, embeddings, backward/optimizer overhead and hardware prevent treating it as exact FLOPs or speedup.
- No claim that changing sharing order alone saves FLOPs or that independently parameterized blocks require extra separate training passes. All are trained jointly on the same unrolled computation.

### Q18: Stick-breaking halting, attention to halted tokens, and ACT

- User requests a from-scratch Section 3.2 explanation and comparison to UT. Crucial notation change: SUT t indexes token positions; l indexes refinement depth. Attention to other time steps means other token positions, subject to attention masks, not future refinement steps.
- Distinguish raw active trajectory h from halt-aware expected representation s; capital S stacks all token representations. Active query reads previous-layer S as K/V; halted tokens remain readable while their own representations stop updating.
- Conditional stop probability q_l (paper hat-alpha) gives unconditional stopping mass alpha_l=q_l*product_{j<l}(1-q_j). Example q=(.2,.5,.75) yields masses (.2,.4,.3), survival .1. Actual stopping uses accumulated mass threshold, not necessarily sampled coins. Expected state at depth3 is .2h1+.4h2+.4h3; survival mass remains on current state. Explain indexing consistently rather than reproducing Algorithm 1's boundary-index ambiguity.
- No-op means identity/do nothing: skip halted token's own expensive update, preserve its readable representation; does not mean remove token from other tokens' attention or eliminate all attention costs.
- Compare UT additive ACT scores and final remainder to SUT survival-product probabilities. SUT criticizes UT's repeated state interpolation as harder to interpret probabilistically and potentially gradient-attenuating, not semantic interpretability of reasoning. Original Graves ACT itself does define probability weights; do not claim otherwise. Both adapt computation depth and penalize excess computation. SUT uses expected depth sum l*alpha_l averaged over tokens, with remaining survival assigned appropriately at a finite cap.
- Sources checked: local PDF Section 3.2 Equations 2-4 and Algorithm 1; original UT appendix; author halting.py. Public ACTWrapper forms conditional two-class softmax stop/continue probabilities, tracks survival and accumulated weighted states, and includes remaining survival mass in current-depth cost. No code changes to experiments.
- Preceding loss discussion: minimize an objective, not necessarily approach zero. Adding a constant preserves ordering and gradients. MSE+10 has minimum10 and remains better when smaller; negative MI has finite lower bound -log(E). Validation and optimization progress, not sign, guide stopping.

Follow the project's AGENTS.md. In particular, preserve performance-first controlled comparisons, prevent label and preprocessing leakage, select checkpoints on validation MSE, distinguish parameter counts from FLOPs, and retain negative results. No experiment changes are authorized or made by this initial reading setup.

### Q19: Worked UT versus SUT halting and ACT comparison

- Use common depth index l, one token, gate outputs q_l; distinguish stopping accumulation from state aggregation. UT Appendix C uses sigmoid-linear gate, additive threshold, final remainder, and recursive interpolation initialized at zero; do not substitute Graves original weighted-sum output for that appendix.
- Teaching example: all gate outputs .5, illustrative threshold .9. UT stops at2, weights(.5,.5), remainder.5, ponder cost2.5. SUT masses(.5,.25,.125,.0625), cumulative .9375 at4. Current-state survival completion yields effective weights(.5,.25,.125,.125), expected capped depth1.875. Prescribed candidate scalar states(10,20,30,40): UT interpolated output12.5; SUT expected state18.75. These artificial shared gate/state values isolate arithmetic and do not imply trained SUT is intrinsically slower or better.
- Finite-tail distinction: paper printed sum l*alpha_l without explicit tail term; example raw sum1.625 omits .0625 surviving mass. Assigning remaining mass to current depth adds .25, yields1.875; public ACTWrapper includes survival contribution. At actual threshold .999 residual is much smaller. Expected depth is not executed count or FLOPs.
- ACT origin checked: Graves Eq10 rho=N+R and Eq12 task+tau*ponder. N discrete; remainder supplies within-region gradient. Moving probability mass from late depth to early depth lowers SUT expected-depth cost directly. Graves footnote1 explicitly discusses survival-product expected cost and a possible mismatch with actual computation; do not imply SUT invented stick-breaking or expected cost, or claim mismatch proves SUT failure.

## Completed presentation — 2026-10-16
- Full 17-slide deck: `docs/slides/sparse-universal-transformer-presentation.pptx`.
- Presenter notes and source links: `docs/slides/sut-presenter-notes.md`.
- Original UT Figure 4, MoA Figure 2, SUT figures, original CFQ construction animation included. Separate MIM, halting/ACT, and state aggregation slides.
- Experiments grouped by translation, CFQ, logical inference; Section 4.4 has its own slide. Approved structural comparison is slide 17, last.
- Table 2 replaces MoMHA with MoA, not FFN MoE. Table 3 concerns FFN experts and selected token associations; it does not quantify redundancy.
- Toy-program results are not included. Review current validated results together before adding slides immediately before the final comparison.
- All 17 exported slides rendered and inspected; final re-import renders match draft previews exactly. Package and layout checks passed, with three heuristic table-height warnings reviewed visually. Native PowerPoint playback/rendering not tested.

## Presentation revision — 2026-10-16
- New deck: `docs/slides/sparse-universal-transformer-revised.pptx`, 19 slides. Prior version preserved.
- UT foundation now precedes VT/UT/SUT comparison. Cover contains authors, affiliations and conference. Figure 1 bottom label corrected with an explicitly adapted caption.
- Equations typeset from LaTeX using MathJax. Editable source: `docs/slides/sut-equations.tex`; LaTeX also retained in equation image descriptions and slide notes. These are rendered equations, not native PowerPoint equation objects.
- Tuples use uppercase E,K,D and E,K,H,D,W with full definitions. Original MoA equations restored, ModuleFormer reference diagram clearly labeled as related architecture rather than an exact MoMHA diagram.
- Loss slide expands original Equation 1 and both entropies. Removed numerical examples. Halting explains N and R; separate UT recurrence and SUT expected-state slides follow it.
- All main experimental tables use original source images, preserving rows, columns and uncertainties. Original CFQ animation shown as two legible static frames for reliable playback-independent display.
- SUT Figure 2 concerns logical inference and composition of NOT/AND. Jump twice is a different SCAN-style task.
- Full revised notes: `docs/slides/sut-presenter-notes-revised.md`. Structural comparison remains the last slide.
