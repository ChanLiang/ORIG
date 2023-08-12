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
bash scripts/train_persona_gpt.sh
bash scripts/train_persona_gpt_kl.sh
```

#### Model inference
```bash
bash scripts/decode_pipeline.sh # for dialogpt
bash scripts/decode_pipeline_naive_gpt.sh # for gpt2
bash scripts/decode_naive_gpt_permutations.sh # decode for all persona permutations
or
python scripts/decdoing.py
```

##### Model evaluation
NLG metrics refer to [nlg-eval](https://github.com/Maluuba/nlg-eval)

The Consistency metric is in [PersonaClassifier](https://github.com/ChanLiang/PersonaClassifier)

Evaluation pipeline:
```bash
bash scripts/eval_pipeline.sh
bash scripts/eval_permutations_pipeline.sh
```

## Citation

```
@misc{chen2023robust,
      title={Towards Robust Personalized Dialogue Generation via Order-Insensitive Representation Regularization}, 
      author={Liang Chen and Hongru Wang and Yang Deng and Wai-Chung Kwan and Zezhong Wang and Kam-Fai Wong},
      year={2023},
      eprint={2305.12782},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
