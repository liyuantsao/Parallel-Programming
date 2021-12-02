make clean 
make

srun -n1 -N1 -c10 ./hw3-1 cases/c16.1 out
rm out