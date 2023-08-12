for((id=0;id<=119;id++))
do
    file=YOUR_FILE_PATH
    echo $id
    wc -l $file

    python run_per_sample.py \
    --ref '../ACL23/decoding_results/ref_testset' \
    --hyp $file
    wait
done

