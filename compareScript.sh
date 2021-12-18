#!/bin/bash

echo "CPU Version"
make jacobi
./jacobi
echo


echo "GPU Version"
make jacobiRWD
srun -N1 --gres=gpu:4 jacobiRWD
echo
