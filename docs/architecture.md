# Hikari Yumiko — ZNC-0 research protocol

## Architecture hypothesis
Experiment with a token-wise gated recurrent state instead of full self-attention.
The candidate computes `p_t = tanh(W_p[x_t;h_(t-1)]+b_p)`,
`g_t = sigmoid(W_g[x_t;h_(t-1)]+b_g)`,
and `h_t = (1-g_t)*h_(t-1) + g_t*p_t`. A projection maps each state to token logits.

**Important:** this is a standard-style gated recurrent mechanism, not a proven new architecture.
We use it as an engineering baseline before researching a novel memory/update operator.

## Current experiment
Synthetic modular arithmetic sequences; supervised next-token prediction with fresh
samples each training step. Reports first and final training losses and held-out
token accuracy. Held-out samples are from the SAME synthetic distribution.
No evidence of human-language understanding, generalization beyond this task,
or reasoning should be inferred.

## Next experiments
- Long-delay key/value recall, with length-stratified held-out sets.
- Separate memory module with configurable capacity.
- Compare against parameter-matched vanilla RNN, GRU, and compact Transformer.
- Log CPU RAM, latency, accuracy, convergence rate and model size.
- Ablation: remove gates, remove external memory, change update schedule.
- Audit existing literature before claiming architectural novelty.
