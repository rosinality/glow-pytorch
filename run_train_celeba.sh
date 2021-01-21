for d in `cat ./parameters/celeba.deltas`
do
    python train.py --delta="$d" `cat ./parameters/celeba.params` --device cuda:0
done
