source /home/rm360179/glow-pytorch/venv/bin/activate
for d in `cat ./parameters/cifar_horses_40.deltas`
do
    srun --partition=common  --qos=16gpu3d --gres=gpu:1 --time 3-0 python train.py --delta="$d" `cat ./parameters/cifar_horses_40.params`  --device cuda:0   &
done
deactivate
