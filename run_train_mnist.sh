for d in 0.005 0.01257 0.03162 0.07953 0.2
do
    python train_mnist.py --batch 128 --img_size 32 --iter 20000 --lr=0.0002 --delta=$d ./mnist/training/
done

