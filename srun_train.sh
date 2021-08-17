#/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Give exactly two arguments:"
    echo "  - a path to your virtualenv, e.g. ./venv"
    echo "  - name of the dataset, e.g. celeba"
    exit 2
fi

activate_path=$1/bin/activate
dataset=$2

source $activate_path
sstatus=$?
if [ "$sstatus" -ne 0 ]; then
    exit 2
fi

for d in `cat ./parameters/$dataset.deltas`
do  
    srun --partition=common  --qos=16gpu3d --gres=gpu:1 --time 3-0 python train.py --delta="$d" `cat ./parameters/$dataset.params`  --device cuda:0 &
done
deactivate
