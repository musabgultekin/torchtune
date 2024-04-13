
[![Unit Test](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtune/actions/workflows/unit_test.yaml)
![Recipe Integration Test](https://github.com/pytorch/torchtune/actions/workflows/recipe_test.yaml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/4Xsdn8Rr9Q?style=flat)](https://discord.gg/4Xsdn8Rr9Q)


# torchtune

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) |  [**Documentation**](https://pytorch.org/torchtune) | [**Design Principles**](#design-principles) | [**Contributing**](#contributing) | [**License**](#license)

&nbsp;

## Introduction

torchtune is a PyTorch-native library for easily authoring, fine-tuning and experimenting with LLMs. We're excited to announce our alpha release!

torchtune provides:

- Native-PyTorch implementations of popular LLMs using composable and modular building blocks
- Easy-to-use, hackable and memory-efficient training recipes for popular fine-tuning techniques (LoRA, QLoRA) - no trainers, no frameworks and < 600 lines of code!
- YAML configs for easily configuring training, evaluation, quantization or inference recipes
- Built-in support for many popular dataset formats and prompt templates to help you quickly get-started with training

torchtune focuses on integrating with popular tools and libraries from the ecosystem. These are just a few examples, with more under development:

- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) for accessing model weights
- [EleutherAI's LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating trained models
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index) for access to training and evaluation datasets
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) for distributed training
- [torchao](https://github.com/pytorch-labs/ao) for lower precision dtypes and post-training quantization techniques
- [Weights and Biases](https://wandb.ai/site) for tracking training progress and logging metrics
- [Executorch](https://pytorch.org/executorch-overview) for on-device inference

&nbsp;

The library currently supports the following models.

| Model                                         | Sizes     |
|-----------------------------------------------|-----------|
| [Llama2](https://llama.meta.com/llama2/)   | 7B, 13B [[models](torchtune/models/llama2/_model_builders.py), [fine-tuning configs](recipes/configs/llama2/)]        |
| [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/)   | 7B [[model](torchtune/models/mistral/_model_builders.py), [fine-tuning configs](recipes/configs/mistral/)]
| [Gemma](https://blog.google/technology/developers/gemma-open-models/)   | 2B [[model](torchtune/models/gemma/_model_builders.py), [fine-tuning configs](recipes/configs/gemma/)]


&nbsp;

---

## Installation

Currently, `torchtune` must be built via cloning the repository and installing.

**Step 1:** Install Pytorch. torchtune is currently tested with the latest stable PyTorch release (2.2.2). We recommend following the instructions [here](https://pytorch.org/get-started/locally/) and installing either version 2.2.2 or the preview nightly version.

**Step 2:** Git clone torchtune and install dependencies.

```bash
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install -e .
```

To confirm that the package is installed correctly, you can run the following command:

```bash
tune --help
```

And should see the following output:

```bash
usage: tune [-h] {ls,cp,download,run,validate} ...

Welcome to the TorchTune CLI!

options:
  -h, --help            show this help message and exit

...
```

&nbsp;

---

## Get Started

To get-started with fine-tuning your first LLM with torchtune, see our tutorial on [fine-tuning Llama2 7B](https://pytorch.org/torchtune/main/examples/first_finetune_tutorial.html). Our [end-to-end workflow](https://pytorch.org/torchtune/main/examples/e2e_flow.html) tutorial will show you how to evaluate, quantize and run inference with this model. The rest of this section will provide a quick overview of these steps with Llama2.

&nbsp;

#### Downloading a model

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama/Llama-2-7b) repository to ensure you have access to the Llama2 model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.

&nbsp;

> Tip: Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access.
You can find your token at https://huggingface.co/settings/tokens

&nbsp;

```bash
tune download meta-llama/Llama-2-7b-hf \
--output-dir /tmp/Llama-2-7b-hf \
--hf-token <HF_TOKEN> \
```

&nbsp;

#### Running fine-tuning recipes

torchtune provides the following fine-tuning recipes.

&nbsp;

> Tip: Single GPU recipes expose a number of memory optimizations that aren't available in the distributed versions. These include support for low-precision optimizers from [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) and fusing optimizer step with backward to reduce memory footprint from the gradients. For memoy-constrained setups, we recommend using the single-device configs as a starting point

&nbsp;


| Training                           | Fine-tuning Method                 | Example Configs                                    |
|------------------------------------|------------------------------------|----------------------------------------------------|
| Distributed Training [1 to 8 GPUs] | [Full](recipes/full_finetune_distributed.py), [LoRA](recipes/lora_finetune_distributed.py)                 | [llama2/13B_full.yaml](), [mistral/7B.lora.yaml]() |
| Single Device / Low Memory [1 GPU] | [Full](recipes/full_finetune_single_device.py), [LoRA and QLoRA](recipes/lora_finetune_single_device.py)       | [llama2/7B_full_low_memory.yaml](recipes/configs/llama2/7B_full_low_memory.yaml), [mistral/7B_qlora_single_device.yaml](recipes/configs/mistral/7B_qlora_single_device.yaml) |
| Single Device [1 GPU]              | [DPO](recipes/lora_dpo_single_device.py)                            | [llama2/7B_lora_dpo_single_device.yaml](recipes/configs/llama2/7B_lora_dpo_single_device.yaml) |


To run a LoRA fine-tune on a single device with llama2 7B using the [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca):
```bash
tune run lora_finetune_single_device --config llama2/7B_lora_single_device
```

torchtune's CLI integrates with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) for easily running distributed training. To run LoRA full-tune on two GPUs with Llama2 7B using the [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)

```bash
tune run --nproc_per_node 2 full_finetune_distributed --config llama2/7B_full_distributed
```

&nbsp;

> Tip: Make sure to place any torchrun commands **before** the recipe specification b/c any other CLI args will
overwrite the config.

&nbsp;

#### Modify Configs

There are two ways in which you can modify configs:

**Config Overrides**

You can easily overwrite config properties from the command-line:

```bash
tune run lora_finetune_single_device \
--config llama2/7B_lora_single_device \
batch_size=8 \
enable_activation_checkpointing=True \
max_steps_per_epoch=128
```

You can also copy the config to your local directory and modify the contents directly:

```bash
tune cp llama2/7B_full .
Copied to ./7B_full.yaml
```

Then, you can run your custom recipe by directing the `tune run` command to your local files:

```bash
tune run full_finetune_distributed --config 7B_full.yaml
```

&nbsp;

Check out `tune --help` for all possible CLI commands and options.

&nbsp;

## Design Principles

torchtune embodies PyTorch’s design philosophy [[details](https://pytorch.org/docs/stable/community/design.html)], especially "usability over everything else".

#### Native PyTorch

torchtune is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: Hugging Face Datasets, EluetherAI Eval Harness), all of the core functionality is written in PyTorch.

#### Simplicity and Extensibility

torchtune is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is prefered over unecessary abstractions
- Modular building blocks over monolithic components

#### Correctness

torchtune provides well-tested components with a high-bar on correctness. The library will never be the first to provide a feature, but available features will be thoroughly tested. We provide

- Extensive unit-tests to ensure component-level numerical parity with reference implementations
- Checkpoint-tests to ensure model-level numerical parity with reference implementations
- Integration tests to ensure recipe-level performance parity with reference implementations on standard benchmarks

&nbsp;

## Community Contributions

We really value our community and the contributions made by our awesome users. We'll use this section to call out some of these contributions!

- [@solitude-alive](https://github.com/solitude-alive) for adding the [Gemma 2B model](torchtune/models/gemma/) to torchtune, including recipe changes, numeric validations of the models and recipe correctness
- [@yechenzhi](https://github.com/yechenzhi) for adding [DPO](recipes/lora_dpo_single_device.py) to torchtune, including the recipe and config along with correctness checks

## Acknowledgements

The Llama2 code in this repository is inspired by the original [Llama2 code](https://github.com/meta-llama/llama/blob/main/llama/model.py).

We also want to give a huge shout-out to EleutherAI, Hugging Face and Weights and Biases for being wonderful collaborators and for contributing the integrations to torchtune. We also want to acknowledge some awesome libraries and tools from the ecosystem:
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast) for performant LLM inference techniques which we've adopted OOTB
- [llama recipes](https://github.com/meta-llama/llama-recipes) for spring-boarding the llama2 community
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for bringing several memory and performance based techniques to the PyTorch ecosystem
- [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) and [lit-gpt](https://github.com/Lightning-AI/litgpt)

&nbsp;

## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

&nbsp;

## License

torchtune is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
