date="2022-11-08"
decoding_strategy="top10_top0.9_T0.9"

# exp_name='baseline'
# exp_name='baseline_shuffle'

cd nlg_metrics

for order in normal_ord pos_ord neg_ord lex_pos_ord lex_neg_ord pos_maj3 pos_maj10 neg_maj3 neg_maj10 
do
    echo $exp_name $order
    # file=../ACL23/decoding_results/${exp_name}_GP2-pretrain-step-7000_${order}_${decoding_strategy}_${date}_pred_response
    file=../ACL23/decoding_results/${exp_name}_GP2-pretrain-step-7000_${order}_${decoding_strategy}_${date}_pred_label
    echo $file
    python run.py \
    --ref '../ACL23/decoding_results/ref_testset' \
    --hyp $file 
    
    wait
done
