# ORIG: Towards Robust Personalized Dialogue Generation via Order-Insensitive Representation Regularization
The implementation for ACL 2023 paper.

The repository is developed based on [Microsoft DialoGPT](https://github.com/microsoft/DialoGPT), [huggingface transformers](https://github.com/huggingface/transfer-learning-conv-ai) and [OpenAI GPT-2](https://github.com/openai/gpt-2).

## Setup & Installation (TL;DR)



#### Environment
Note: The script below may not be sufficient and missing packages need to be configured manually.
```bash
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

## Pipeline details

#### Training script
```bash
bash train_persona_gpt.sh
bash train_persona_gpt_kl.sh
```

#### Model inference
```bash
bash decode_pipeline.sh # for dialogpt
bash decode_pipeline_naive_gpt.sh # for gpt2
bash decode_naive_gpt_permutations.sh # decode for all persona permutations
or
python decdoing.py
```

##### Model evaluation
NLG metrics refer to [nlg-eval](https://github.com/Maluuba/nlg-eval)

The Consistency metric is in [PersonaClassifier](https://github.com/ChanLiang/PersonaClassifier)

Evaluation pipeline:
```bash
bash eval_pipeline.sh
bash eval_permutations_pipeline.sh
```
