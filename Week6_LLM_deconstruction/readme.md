# Week 6: Deconstructing an Open-Source LLM

##  Objectives
* Bridge the gap between human string data and tensor geometry via Tokenization and Embedding tables.
* Understand the architectural deviations that make Llama 3 and Mistral 7B computationally viable.
* Implement Root Mean Square Normalization (RMSNorm).

##  The Data Translation Pipeline
Machine learning models cannot read text. To process language, we use a two-step analog-to-digital conversion:
1. **Tokenization (BPE):** A structural algorithm chunks strings into optimal sub-word IDs. "Unbelievable" might become `[Un, believ, able]`. 
2. **Embeddings:** These discrete IDs index a massive `nn.Embedding` table, pulling out a dense, high-dimensional vector. Words with similar semantic meanings are mathematically pushed closer together in this geometric space.

##  Modern Architectural Upgrades
When building models with 7B+ parameters, the vanilla 2017 Transformer requires too much memory and compute. Modern models utilize the following upgrades:

* **RMSNorm vs. LayerNorm:** Layer Normalization calculates the mean and variance to center and scale activations. RMSNorm removes the mean-centering step entirely. By assuming the mean is near zero, it saves massive computational overhead while maintaining gradient stability.
  $$RMS(x) = \sqrt{\frac{1}{n} \sum x_i^2}$$
  $$y = \frac{x}{RMS(x)} \cdot \gamma$$

* **RoPE (Rotary Positional Embeddings):** Instead of adding a static position vector to the embeddings, RoPE applies a geometric rotation to the Query and Key vectors in the complex plane. This allows the attention mechanism to naturally understand the *relative* distance between tokens, which scales perfectly to longer context windows.

* **GQA (Grouped-Query Attention):** Instead of every Query head having its own Key and Value head, GQA clusters multiple Queries to share a single KV pair. This drastically reduces the size of the KV Cache held in memory during the inference generation loop.

##  Usage
To run the text-to-tensor pipeline and verify the variance stabilization of RMSNorm, execute:

```bash
python llm_components.py
