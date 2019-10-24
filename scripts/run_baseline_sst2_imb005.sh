for t in {1..15}
do
    python -u baseline_main.py \
        --task sst-2 \
        --train_num_per_class 1000 \
        --imbalance_rate 0.05 \
        --dev_num_per_class 10 \
        --epochs 10 \
        --batch_size 25
done
