


# ckp_path='output/persona/DialoGPT-naive_gpt2_base_single_turn-lr-2e-05-bz-64-time-2022-12-11000903/GP2-pretrain-step-1000.pkl'
ckp_path="output/persona/DialoGPT-naive_gpt2_base-lr-2e-05-bz-64-time-2022-12-10003447/GP2-pretrain-step-1500.pkl"
w_typeId=False
eos_in_decoding=False
input_ground_truth_persona_label=False

# date="2022-12-12"
# date="2022-12-26"
date="2023-01-12"
beam=5
# beam=10
# min_decode_length=10
# min_decode_length=20

# batch_size=32 # normal seq
batch_size=16 # normal seq
# batch_size=8 # long seq, beam search
# batch_size=16 # long seq, beam search
# batch_size=1 # debug

debug=False
# debug=True

# gpu=2
# gpu=3
gpu=1

order=normal_ord

# for decoding_strategy in "beam_search" "top10_top0.9_T0.9"
for decoding_strategy in "top10_top0.9_T0.9"
do
echo $decoding_strategy
# for((id=0;id<=30;id++))
# for((id=31;id<=60;id++))
# for((id=61;id<=75;id++))
for((id=76;id<=90;id++))
# for((id=0;id<=0;id++))
do
    exp_name=naive_gpt2-small_permutation_${id}
    echo $exp_name

    python decoding.py \
    --date $date \
    --exp_name $exp_name \
    --model_path $ckp_path \
    --order $order \
    --decoding_strategy $decoding_strategy \
    --beam $beam \
    --eos_in_decoding $eos_in_decoding \
    --input_ground_truth_label $input_ground_truth_persona_label \
    --batch_size $batch_size \
    --debug $debug \
    --gpu $gpu \
    --w_typeId $w_typeId \
    --permutaion_id ${id} \
    --bar True
    wait
done

done