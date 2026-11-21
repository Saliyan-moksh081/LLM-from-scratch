# Build and Train an LLM

This project will build a small decoder-only language model in Python and PyTorch, train it from randomly initialized weights, and generate text from its checkpoints. The aim is to understand and own the training pipeline—not to fine-tune an existing foundation model or claim frontier-model quality.

Development will happen locally. Training jobs may run on a shared NVIDIA RTX A6000 server, but only within a GPU memory and runtime budget approved for its production workloads.

## Goals

- Implement and understand the core pieces of a language model and its training loop.
- Start with a small, end-to-end run: prepare text, tokenize it, train a model, save a checkpoint, and generate text.
- Keep the code Python-first, readable, testable, and small enough to understand.
- Use [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) as the primary implementation reference. Before writing a component, inspect its corresponding source and adapt the simplest relevant approach. Write new code only when the reference does not cover the requirement, following the minimal-code guardrails in the project instructions.
- Develop and test locally; run approved training jobs remotely; keep checkpoints, logs, and configuration explicit.

## Initial scope

The first milestone is a **small base language model trained from scratch** on a modest text corpus. It will learn next-token prediction and produce text continuations. Instruction tuning, preference optimization, reinforcement learning, a web UI, and distributed training are later possibilities, not part of the first milestone.

The first milestone is complete when we can:

1. Prepare a documented text dataset and convert it into token IDs.
2. Run a short training job that shows the training and validation loss.
3. Save and reload a checkpoint, including the model configuration and tokenizer details needed to use it.
4. Generate a text continuation from the saved checkpoint.
5. Run a small test suite locally and a short, explicitly approved GPU smoke test remotely.

The dataset, tokenizer, initial model dimensions, and production-safe GPU budget will be chosen before the training implementation. The first run will be deliberately small; model size and training duration will be increased only after measuring results and resource use.

## Planned architecture

```text
Public or user-provided text
          │
          ▼
Data preparation ──► Tokenizer ──► Token-ID training/validation data
                                          │
                                          ▼
                               Decoder-only Transformer
                                          │
                                          ▼
                         PyTorch next-token training loop
                                          │
                                          ├──► Checkpoints and training logs
                                          ▼
                             Evaluation and text generation
                                          │
                               optional, later step
                                          ▼
                              vLLM or Ollama serving
```

### Main parts

- **Data preparation:** load the selected corpus, apply only necessary cleaning, make a stable train/validation split, and prepare data for tokenization. Dataset source, license, and preparation steps will be recorded.
- **Tokenizer:** turn text into integer token IDs and preserve the tokenizer configuration with the checkpoint. We will choose whether to use the reference project's tokenizer approach or its small from-scratch BPE implementation based on the first dataset and learning goals.
- **Model:** begin with a conventional causal, decoder-only Transformer implemented in PyTorch. Each position predicts the next token, and causal attention prevents it from seeing future tokens. The reference implementation is in [`src/models/transformer.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/models/transformer.py) and its component modules under [`src/models/`](https://github.com/FareedKhan-dev/train-llm-from-scratch/tree/main/src/models).
- **Training:** sample token sequences, compute next-token cross-entropy loss, backpropagate, and periodically evaluate and save resumable checkpoints. The reference training paths are [`scripts/train_transformer.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/train_transformer.py) for the simple path and [`scripts/pretrain_base.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/pretrain_base.py) for the more capable path. We will start with the simplest one that meets the first milestone.
- **Evaluation and generation:** compare train and validation loss, reload a checkpoint, and sample continuation tokens to verify the full pipeline.
- **Serving (later):** vLLM or Ollama can expose a trained, compatible checkpoint for inference. They are serving tools, not the mechanism that runs the PyTorch training loop.

### Keep the first code layout small

The initial implementation should use only the files needed for the first working pipeline. A likely starting shape is:

```text
README.md
pyproject.toml
llm/
  model.py
  data.py
  train.py
  generate.py
tests/
```

This is a proposal, not a requirement to create all of these files at once. We will add a file only when it makes the working implementation clearer. Training code and project logic will be Python; standard project metadata, configuration, and documentation may use their conventional formats.

## Development and training workflow

1. **Develop locally:** implement one small part at a time, compare it with the reference repository, and run focused CPU tests where practical.
2. **Smoke-test locally:** use tiny settings and a small dataset to validate data shapes, loss calculation, checkpoint save/load, and generation without occupying the production GPU.
3. **Prepare the server job:** sync a reviewed commit to the server, check current GPU use and the agreed resource budget, and use a small batch/context configuration for the first GPU test.
4. **Train conservatively:** log the configuration and resource use, save checkpoints and logs to known locations, and stop if the production service needs the capacity or is affected.
5. **Review before scaling:** use measured throughput, memory, validation loss, and production headroom to decide whether the next run can safely grow.

The current `nvidia-smi` screenshot is only a point-in-time observation: it showed 18,762 MiB used out of 49,140 MiB, with two vLLM processes using most of that allocation. The remaining memory is **not** automatically a training budget; the serving workload can change, and production stability takes priority. A safe memory ceiling and training window must be agreed with the server owner before running jobs. Low GPU utilization at one moment does not guarantee that training is safe.

## Roadmap

1. **Confirm the experiment:** choose the intended text/domain, dataset and license, tokenizer, and a safe server budget.
2. **Build the smallest working baseline:** data loading, a basic decoder-only Transformer, next-token training, validation loss, and checkpoints.
3. **Complete the end-to-end path:** reload the checkpoint and generate text; document exact local and remote commands.
4. **Measure and improve:** compare model/data sizes and training settings within the approved resource budget.
5. **Consider post-training:** only after the base model works, evaluate whether SFT or other alignment stages are useful for the project's goal.
6. **Consider serving:** use vLLM or Ollama for inference if the resulting architecture and checkpoint format are supported and serving capacity is available.

## Reference project

Primary reference: [FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch). It contains an educational Transformer implementation, tokenization and data preparation, pretraining scripts, generation, evaluation, and optional post-training stages. Its license is MIT. We will consult the relevant upstream source for each implementation and retain required license notices when reusing code. The reference's configurations and hardware estimates are examples, not guaranteed settings for this shared server.

## Out of scope for the first version

- Training a frontier-scale model or promising production-grade answer quality.
- Starting with a pretrained model or building a fine-tuning-only project.
- Running training through Ollama or vLLM APIs.
- Launching long jobs or using all apparently free GPU memory without an explicit production-safe budget.
- Adding distributed training, complex experiment platforms, or a UI before the basic training-and-generation path works.
