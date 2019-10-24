for t in {1..15}
do
    python -u ren_main.py \
        --task sst-5 \
        --train_num_per_class 40 \
        --dev_num_per_class 2 \
        --batch_size 8 \
        --epochs 10 \
        --pretrain_epochs 3 \
        --min_epochs 1
done
