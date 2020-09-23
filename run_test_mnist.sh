for d in `cat ./parameters/mnist.deltas`
do
    python train.py --delta="$d" `cat ./parameters/mnist.params` ./checkpoint/model_"$d"_.pt
done
