for t in {1..15}
do
    python -u weighting_main.py \
        --task sst-2 \
        --train_num_per_class 1000 \
        --imbalance_rate 0.05 \
        --dev_num_per_class 10 \
        --epochs 10 \
        --batch_size 25 \
        --w_init 1. \
        --w_decay 5. \
        --norm_fn linear
done

# (batch_size=25, data_seed=159, dev_num_per_class=10, epochs=10, imbalance_rate=0.05, learning_rate=4e-05, min_epochs=0, norm_fn='linear', pretrain_epochs=0, task='sst-2', train_num_per_class=1000, w_decay=5.0, w_init=1.0)