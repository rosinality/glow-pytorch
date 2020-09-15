for d in 0.01000 0.01292 0.01668 0.02154 0.02783 0.03594 0.04642 0.05995 0.07743 0.10000
do
    python train.py --batch 32 --img_size 16 --iter 30000 --lr=0.00005 --delta=$d ./mnist/training/
done

