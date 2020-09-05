for t in {1..15}
do
    python -u baseline_main.py \
        --task sst-5 \
        --train_num_per_class 40 \
        --dev_num_per_class 2 \
        --batch_size 8 \
        --epochs 10
done
