# ===========================================================================
# 1(a)【joint decoding exp parameters】
# exp_name='joint_decoding'
# ckp_path='output/persona/DialoGPT-w_persona_label_eos_response_unshuffle-lr-1e-05-bz-32-time-2022-10-02142905/GP2-pretrain-step-7000.pkl'
# eos_in_decoding=True
# input_ground_truth_persona_label=False
# gpu=2

# exp_name='joint_decoding_w_groundtruth_label'
# ckp_path='output/persona/DialoGPT-w_persona_label_eos_response_unshuffle-lr-1e-05-bz-32-time-2022-10-02142905/GP2-pretrain-step-7000.pkl'
# eos_in_decoding=False
# input_ground_truth_persona_label=True
# gpu=0

# exp_name='joint_decoding_wo_personaThreshold'
# ckp_path='output/persona/DialoGPT-w_persona_label_wo_threshold_eos_response_shuffle-lr-1e-05-bz-32-time-2022-11-17005337/GP2-pretrain-step-7000.pkl'
# w_typeId=True
# eos_in_decoding=True
# input_ground_truth_persona_label=False
# gpu=1

# ===========================================================================
# 1(b)【baseline exp parameters】
# exp_name='baseline'
# ckp_path='output/persona/DialoGPT-wo_persona_label-lr-1e-05-bz-32-time-2022-10-01132438/GP2-pretrain-step-7000.pkl'
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=2

# exp_name='baseline_shuffle'
# ckp_path='output/persona/DialoGPT-wo_persona_label_shuffle-lr-1e-05-bz-32-time-2022-10-01132251/GP2-pretrain-step-7000.pkl'
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=1

# exp_name='baseline_wo_typeId'
# ckp_path='output/persona/DialoGPT-baseline_wo_typeId-lr-1e-05-bz-32-time-2022-11-09003915/GP2-pretrain-step-7000.pkl'
# w_typeId=False
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=2

# exp_name='baseline_wo_typeId_all_seq_loss'
# ckp_path='output/persona/DialoGPT-baseline_wo_typeId_w_all_loss-lr-1e-05-bz-32-time-2022-11-09115342/GP2-pretrain-step-2000.pkl'
# w_typeId=False
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=1

# exp_name='naive_dialogpt_small'
# ckp_path='output/persona/DialoGPT-naive_dialogpt_base-lr-2e-05-bz-64-time-2022-12-09192220/GP2-pretrain-step-1000.pkl'
# w_typeId=False
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=2

# exp_name='naive_gpt2-small'
# ckp_path='output/persona/DialoGPT-naive_gpt2_base-lr-2e-05-bz-64-time-2022-12-10003447/GP2-pretrain-step-1500.pkl'
# w_typeId=False
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=2

# exp_name='naive_gpt2-small_single_turn'
# ckp_path='output/persona/DialoGPT-naive_gpt2_base_single_turn-lr-2e-05-bz-64-time-2022-12-11000903/GP2-pretrain-step-1000.pkl'
# w_typeId=False
# eos_in_decoding=False
# input_ground_truth_persona_label=False
# gpu=2

exp_name='naive_gpt2-small_attention_lam10'
ckp_path='output/persona/DialoGPT-naive_gpt2_base_attention_lam10-lr-2e-05-bz-64-time-2022-12-23093616/GP2-pretrain-step-1500.pkl'
w_typeId=False
eos_in_decoding=False
input_ground_truth_persona_label=False
model_size='small'
# gpu=2

# ===========================================================================
# 2.【general parameters】
# date="2022-10-12"
# date="2022-11-08"
# date="2022-11-10"

# date="2022-12-09"
# date="2022-12-10"
# date="2022-12-12"
# date="2022-12-26"
date="2022-12-27"

# decoding_strategy="top10_top0.9_T0.9"
# decoding_strategy="beam_search"
beam=5
# beam=10
# min_decode_length=10
# min_decode_length=20

batch_size=128 # Small model: to fill up the memory...
batch_size=64 # Medium model: to fill up the memory...
# batch_size=32 # normal seq
# batch_size=8 # long seq, beam search
# batch_size=16 # long seq, beam search
# batch_size=1 # debug

debug=False
# debug=True

gpu=2

# ===========================================================================
# 3.【decoding shell】
# for order in normal_ord
# for order in normal_ord pos_ord neg_ord
# for order in pos_ord
# for decoding_strategy in "beam_search" "top10_top0.9_T0.9"
for decoding_strategy in "top10_top0.9_T0.9"
do
echo $decoding_strategy
for order in normal_ord
# for order in normal_ord pos_ord neg_ord lex_pos_ord lex_neg_ord 
# for order in normal_ord pos_ord neg_ord lex_pos_ord lex_neg_ord pos_maj3 pos_maj10 neg_maj3 neg_maj10 
    do
        echo $order
        python decoding.py \
        --date $date \
        --exp_name $exp_name \
        --model_path $ckp_path \
        --model_size $model_size \
        --order $order \
        --decoding_strategy $decoding_strategy \
        --beam $beam \
        --eos_in_decoding $eos_in_decoding \
        --input_ground_truth_label $input_ground_truth_persona_label \
        --batch_size $batch_size \
        --debug $debug \
        --gpu $gpu \
        --w_typeId $w_typeId \
        --bar True
        wait
    done

done