source /home/ptempczyk/virtualenvs/glow-pytorch/bin/activate
python ./train_delta_sequence.py `cat ./parameters/mnist.params` --device cuda:0 &
python ./train_delta_sequence.py `cat ./parameters/fashion.params` --device cuda:0 &
deactivate