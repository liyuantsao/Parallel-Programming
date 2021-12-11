make clean
make

# srun -n1 -N1 --gres=gpu:1 ./hw3-2 cases/c08.1 out
./hw3-2 cases/c04.1 out

# rm out