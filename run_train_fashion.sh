for d in `cat ./parameters/fashion.deltas`
do
    python train.py --delta="$d" `cat ./parameters/fashion.params`
done
