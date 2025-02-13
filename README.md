# MBR-Alignment
In this project we leverage recent advances in Large Language Models (LLM) preference
optimisation to align the LLM using a preference set generated through Minimum Bayes risk
(MBR) decoding. MBR decoding is a two pass decoding method, that effectively addresses
some of the weaknesses of commonly used single pass decoding methods. MBR decoding
has shown to enhance the performance and diversity of the generated outputs compared to
single pass decoding methods. Yet MBR’s superior performance comes at a cost, as a it
scales quadratically with the number of samples. This renders MBR impractical to use at
inference time.
Unsupervised preference learning using a preference set generated through MBR
enables the model to learn the MBR preferences, and match the MBR performance through a
single pass decoding algorithm. We validate the efficacy of this method on three different
tasks: Question Answering on StrategyQA, Summarization on CNN/DM, and Question
Generation on SquadV2. To align the models, we use Direct Preference Optimization (DPO)
and Kahneman-Tversky Optimization (KTO). We investigate the performance of the chosen
alignment methods with different degrees of regularization and different preference set
generation strategies. We find that DPO achieves on all three tasks the best performance.
Guided by MBR data properties, we explain why DPO outperforms KTO.
Our work shows the efficacy of unsupervised preference learning using MBR data, as it
works across tasks and alignment techniques.
