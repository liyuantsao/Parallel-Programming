make clean
make

srun -n1 -N1 -c2 --gres=gpu:2 ./hw3-3 cases/c05.1 out
# ./hw3-3 cases/c05.1 out

# rm out