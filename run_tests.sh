for d in 0.01 0.01292
do
    python test.py --batch 32 --img_size 16 --iter 6 --lr=0.00005 --delta=$d ./checkpoint/model_"$d"_.pt ./mnist/training/
done

