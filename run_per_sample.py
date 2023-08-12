import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ref", type=str, default="persona")
parser.add_argument("--hyp", type=str, default="persona")

args = parser.parse_args()

try:
    from metrics.compute_metrics_per_sample import compute_metrics
except:
    from src.style_rl.metrics.compute_metrics import compute_metrics
import logging
logger = logging.getLogger(__name__)
import numpy as np

refs = []
hyps = []
r1 = open(args.ref, encoding='utf-8', mode='r')
r2 = open(args.hyp, encoding='utf-8', mode='r')
w = open(args.hyp + '_metrics', encoding='utf-8', mode='w')
for tgt, hypo in zip(r1, r2):
    refs.append([tgt])
    hyps.append(hypo)
metrics = compute_metrics(hyps, refs, no_emb=True) # return a dic
metrics['avg_length'] = np.mean([len(_.split()) for _ in hyps])
# print(metrics)

# for m, s in metrics.items():
#     # print(f'{m}:  {s:.4f}')
#     if 'list' in m:
#         continue
#     if s < 1:
#         s = s * 100
#         print (f'{m}:  {s:.2f}')
#     else:
#         print(f'{m}:  {s:.4f}')

# for m in ['bleu-1_list', 'rouge_list', 'distinct_1_list']:
#     print (f'{m} = {np.mean(metrics[m])}')

for i in range(len(refs)):
    # w.write(f"{metrics['bleu-1_list'][i]}\t{metrics['rouge_list'][i]}\t{metrics['distinct_1_list'][i]}\n")
    w.write(f"{metrics['rouge_list'][i]}\n")