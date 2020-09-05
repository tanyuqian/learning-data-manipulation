for t in {1..15}
do
    python -u weighting_main.py \
        --task sst-5 \
        --train_num_per_class 40 \
        --dev_num_per_class 2 \
        --batch_size 8 \
        --epochs 10 \
        --pretrain_epochs 3 \
        --min_epochs 1 \
        --w_init 1. \
        --w_decay 5. \
        --norm_fn softmax
done

# (batch_size=8, data_seed=159, dev_num_per_class=2, epochs=10, imbalance_rate=1.0, learning_rate=4e-05, min_epochs=1, norm_fn='softmax', pretrain_epochs=3, task='sst-5', train_num_per_class=40, w_decay=5.0, w_init=1.0)