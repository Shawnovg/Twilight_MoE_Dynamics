# Halting and state comparison: three replacement slides

## 1. Dynamic halting: addition versus stick breaking

Suggested spoken explanation: Both models learn when to stop refining a token. UT adds gate scores. SUT interprets its gate as taking a fraction of the probability mass still available, like repeatedly breaking the remaining part of a stick. Once enough mass has been assigned to stopping, the token stops updating. ACT encourages earlier stopping.
Use one-token notation throughout. q_l is a gate output in [0,1]. C_0=0. SUT begins with m_0=1; p_l=q_l m_(l-1) is unconditional stopping probability and m_l=(1-q_l)m_(l-1) is survival mass. Therefore p_l=q_l product_(j<l)(1-q_j), exactly the stick-breaking algebra in SUT Equation 2 with q=alpha-hat and p=alpha. Cumulative SUT stopping mass is 1-m_l. UT uses cumulative q instead. The two learned gates need not have equal values or identical parameterizations; UT uses a sigmoid-linear gate, SUT an MLP.
UT N is the first depth at which the cumulative q exceeds its threshold, subject to a depth cap. Its last interpolation update uses R=1-sum_(j<N)q_j. This remainder fills to one, not merely to the threshold. R is distinct from SUT survival mass m. The per-token ACT ponder cost is N+R. SUT Equation 4 is expected stopping depth, sum_l l p_l. Average per-token costs over token positions and multiply by the training coefficient in the total loss. The full SUT objective is L_ACT=(1/T)sum_i sum_l l alpha_l^(i). The public ACTWrapper includes a surviving-mass-at-current-depth term for finite-depth accounting. These training penalties are not measured FLOPs or wall-clock runtime.
Threshold values are omitted to keep the distinction visible; SUT paper uses .999. The strip illustration is a newly generated conceptual teaching aid, not an original paper figure or an empirical chart. Segment lengths are illustrative only. White area on the third strip denotes mass already assigned at a previous step, dark green newly assigned mass, pale green surviving mass.

Sources:
https://arxiv.org/pdf/1807.03819v3
https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX ut_acc: C_{\ell}=C_{\ell-1}+q_{\ell}

LaTeX sut_break: p_{\ell}=q_{\ell}m_{\ell-1},\qquad m_{\ell}=(1-q_{\ell})m_{\ell-1}

LaTeX ut_act: \rho_{\mathrm{UT}}=N+R,\qquad R=1-\sum_{j=1}^{N-1}q_j

LaTeX sut_act: \rho_{\mathrm{SUT}}=\sum_{\ell=1}^{L}\ell\,p_{\ell}

## 2. Hidden states before and after halting

Suggested spoken explanation: First distinguish what a block computes from what halting combines. x_l is a token's complete block output before mixing across depths. s_l is the weighted state produced by halting. They have the same embedding dimension but different roles. This prevents H, h and s from being mistaken for the same tensor.
We deliberately use new aligned symbols x_l and s_l, rather than pretend the two papers use identical notation. The token index i is suppressed unless needed. UT H^t in Equation 4 is a matrix of token states after the complete update; its t denotes depth. One row maps to our x_l. UT Appendix C calls the raw block output transformed_state and the accumulated output new_state/previous_state. In SUT, x_l maps to h_l^(i), and s_l maps to s_l^(i). Capital S stacks the s states of all token positions. The character l is lowercase ell for refinement depth, not capital I. The state vectors are model-width representations before Q/K/V projection, not attention-score vectors.
Normalization answer: in original UT Figure 4 and Equation 4, the block output follows the final transition/FFN residual addition and LayerNorm. In the SUT code, x is the value returned by sut_transformer_layer.forward before ACTWrapper.halt aggregation. The exact last operation is configuration-dependent: post-norm blocks return after the last LayerNorm; pre-norm blocks return after the residual addition. The SUT Figure 3 schematic shows pre-norm placements. Thus it is incorrect to label every SUT hidden state as after a final LayerNorm without specifying the setting. A possible extra stack-level normalization after all recurrent steps is separate from this per-block output.
Both networks reuse block parameters over depth. That recurrence is distinct from combining outputs through the halting rule. Original UT Figure 4 encoder crop and SUT Figure 3 left are used to locate these states.
Code: https://github.com/shawntan/SUT/blob/main/sut_layer/sut_transformer_layer.py
https://github.com/shawntan/SUT/blob/main/halting.py
https://github.com/shawntan/SUT/blob/main/transformer_encoder.py

Sources:
https://arxiv.org/pdf/1807.03819v3
https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX raw: \mathbf{x}_{\ell}\quad\text{block output before halting aggregation}

LaTeX mixed: \mathbf{s}_{\ell}\quad\text{halting-weighted state}

## 3. State aggregation: compare the same quantities

Suggested spoken explanation: Now both columns use x for the new block output and s for the combined output. UT combines the new x with the previous mixture s. SUT combines the surviving current x with explicitly weighted earlier x states. This is the difference in aggregation, not the fact that the block repeats. In SUT, an active query can read the combined states of other tokens, even if they have halted.
UT formula comes from Appendix C new_state = update_weights * transformed_state + (1-update_weights) * previous_state. w_l is q_l before the last step, the remainder R on the last step, and zero after stopping. SUT Equation 3 becomes s_l=m_(l-1)x_l+sum_(j<l)p_j x_j with m_(l-1)=1-sum_(j<l)p_j. This uses exactly the original strict j<l summation, avoiding an off-by-one interpretation that would count the current stopping probability twice. p_j=q_j product_(k<j)(1-q_k).
UT later factors (1-w_l) multiply the previous mixture, so older contributions are repeatedly rescaled. SUT exposes earlier stopping weights directly. Do not describe the original Graves ACT weighted sum as identical to UT Appendix C interpolation. UT raw transformed states, not merely its accumulated output, feed the next raw update in the Appendix C loop.
In SUT Algorithm 1 the active query representation is raw h_(l-1)^(i) (our x), whereas K and V use S_(l-1), the matrix of combined states. Q/K/V here name inputs to projections, not already projected vectors. For a halted token, x and s are copied/no-op and remain available to other tokens, subject to attention masks. s need not itself be a LayerNorm output; it is a mixture of block outputs. This corrects the ambiguity in the previous three slides.
The SUT paper has indexing edge cases around its pseudocode initialization; here we preserve Equation 3 and explain its state roles, rather than reproduce the first-iteration pseudocode literally.

Sources:
https://arxiv.org/pdf/1807.03819v3
https://aclanthology.org/2023.emnlp-main.12.pdf

LaTeX ut_mix: \mathbf{s}_{\ell}=\underbrace{w_{\ell}\mathbf{x}_{\ell}}_{\text{current block output}}+\underbrace{(1-w_{\ell})\mathbf{s}_{\ell-1}}_{\text{previous mixture}}

LaTeX sut_mix: \mathbf{s}_{\ell}=\underbrace{m_{\ell-1}\mathbf{x}_{\ell}}_{\text{current surviving mass}}+\underbrace{\sum_{j<\ell}p_j\mathbf{x}_j}_{\text{earlier stopping contributions}}

LaTeX weights: w_{\ell}=q_{\ell}\;(\ell<N),\qquad w_N=R

LaTeX survival: m_{\ell-1}=1-\sum_{j<\ell}p_j

LaTeX attention: Q\leftarrow\mathbf{x}_{\ell-1}^{(i)},\qquad K,V\leftarrow\mathbf{S}_{\ell-1}
