for t in {1..30}
do
  python -u baseline_main.py \
    --task cifar-10 \
    --data_seed 159 \
    --epochs 15 \
    --train_num_per_class 40 \
    --dev_num_per_class 2 \
    --resnet_pretrained \
    --batch_size 10
done
