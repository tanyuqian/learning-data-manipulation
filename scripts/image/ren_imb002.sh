for t in {1..30}
do
  python -u ren_main.py \
    --task cifar-10 \
    --data_seed 159 \
    --epochs 15 \
    --batch_size 20 \
    --train_num_per_class 1000 \
    --dev_num_per_class 10 \
    --resnet_pretrained \
    --imbalance_rate 0.02

done
