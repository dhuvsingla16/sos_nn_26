# Summer of Science (SoS) 2026: Neural Networks & Deep Learning

**Author:** Dhruv Singla  
**Roll Number:** 24B1211  
**Mentor:** Alok Kale  

##  About This Repository
This repository tracks my continuous progress, codebase, and mathematical derivations for the Summer of Science (SoS) 2026 program. The core focus of this project is to build a rock-solid, intuitive understanding of Neural Networks and Deep Learning by intentionally "reinventing the wheel."

Instead of immediately relying on high-level frameworks, this project starts from the absolute ground up. It begins with manual calculus and pure NumPy matrix operations to build Multi-Layer Perceptrons from scratch, before transitioning into PyTorch to deconstruct the mathematics of Self-Attention, Large Language Model (LLM) architectures, and Parameter-Efficient Fine-Tuning (PEFT).

##  Repository Structure
The repository is organized chronologically by week. **Each folder contains its own dedicated `README.md`** that thoroughly explains the mathematical concepts, architectural decisions, and execution instructions for that specific week's actionable task.

* **`Week-1/`** — Core Concepts & Single Neuron Math (Building a scalar automatic differentiation engine).
* **`Week2_backprop_math/`** — Backpropagation & Optimization (The Chain Rule and topological sorting).
* **`Week3_Evaluation_basics/`** — Bias, Variance, and Data Splitting (Manual confusion matrix and metrics calculation).
* **`Week4_numpy_mlp_midterm/`** — Ground-Up Implementation (A fully vectorized NumPy MLP with L2 Regularization).
* **`Week5_pytorch_attention/`** — PyTorch Transition & The Math of Causal Self-Attention.
* **`Week6_LLM_deconstruction/`** — Text to Tensors, RMSNorm, and Modern LLM Architecture Upgrades.
* **`Week7_peft_math/`** — The Mathematics of Parameter-Efficient Fine-Tuning (LoRA and Quantization concepts).
* **`midterm_report_v3.pdf`** — A comprehensive LaTeX report detailing the math, code, and visualizations for the first four weeks of the project.

##  How to Navigate
To follow the learning progression, start at `Week-1` and move sequentially. Open the `README.md` inside any specific folder to explore the exact equations, code breakdowns, and terminal commands needed to run that week's Python scripts.
