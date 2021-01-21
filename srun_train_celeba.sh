source /home/ptempczyk/virtualenvs/glow-pytorch/bin/activate
for d in `cat ./parameters/celeba.deltas`
do
    srun --partition=common --qos=16gpu3d --gres=gpu:1 python train.py --delta="$d" `cat ./parameters/celeba.params` --device cuda:0 &
done
deactivate