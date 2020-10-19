source /home/ptempczyk/virtualenvs/glow-pytorch/bin/activate
srun --partition=common --qos=8gpu7d --gres=gpu:1 python ./train_delta_sequence.py `cat ./parameters/mnist.params` --device cuda:0 &
srun --partition=common --qos=8gpu7d --gres=gpu:1 python ./train_delta_sequence.py `cat ./parameters/fashion.params` --device cuda:0 &
deactivate