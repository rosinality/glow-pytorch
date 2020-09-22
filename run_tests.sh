for d in 0.01 0.01292 0.01668 0.02154 0.02783 0.03594 0.04642 0.05995
do
    python test.py --batch 32 --img_size 16 --iter 60000 --lr=0.00005 --delta=$d ./old/mnist_60k/checkpoint/model_"$d"_.pt ./mnist/training/
done

