# Week 7: The Mathematics of Parameter-Efficient Fine-Tuning (PEFT)

##  Objectives
* Understand the memory bottlenecks of full fine-tuning.
* Implement the matrix decomposition math behind LoRA (Low-Rank Adaptation).
* Understand how Quantization compresses neural network weights to run on consumer hardware (QLoRA).

##  The Mathematics of LoRA
When fine-tuning a neural network, the traditional approach updates the entire weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by adding a gradient update matrix $\Delta W$ of the exact same size:
$$W_{new} = W_0 + \Delta W$$

If $W_0$ has 100 million parameters, backpropagation must track 100 million gradients and 200 million optimizer states. This explodes VRAM requirements.

The LoRA paper hypothesis is that the "intrinsic rank" of the changes needed for a specific task is actually very low. Instead of calculating a full $\Delta W$, we freeze $W_0$ and decompose the update into two much smaller matrices, $A$ and $B$:
$$\Delta W = B A$$
Where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$, and the rank $r$ is a very small number (e.g., 4, 8, or 16).

The forward pass equation becomes:
$$h = W_0 x + \frac{\alpha}{r} B A x$$

* **Matrix A:** Down-projects the input space. Initialized with random variance.
* **Matrix B:** Up-projects back to the output space. Initialized to $0$ so the initial training step doesn't corrupt the base model.
* **$\alpha$ (Alpha):** A scaling constant that dictates how strongly the LoRA adapter influences the base model.

##  Quantization (QLoRA)
Even if we are only training a small adapter, the massive frozen base model still needs to sit in GPU memory. Quantization solves this by casting the standard 32-bit floating-point weights (FP32) into 8-bit or 4-bit integers (INT8/NF4). This reduces the memory footprint of the base model by 4x to 8x, allowing massive models to fit onto a single GPU while the high-precision LoRA adapters handle the actual learning.

##  Usage
To run the parameter reduction calculation, verify the LoRA forward pass, and test the conceptual quantization memory savings, execute:

```bash
python lora_math.py
