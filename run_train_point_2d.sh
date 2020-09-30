for d in `cat ./parameters/point_2d.deltas`
do
    python train.py --delta="$d" `cat ./parameters/point_2d.params`
done
