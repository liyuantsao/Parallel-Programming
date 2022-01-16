make
g++ -o judge judge.cc
rm output/*.out
# srun -N2 -c5 ./hw4 TEST01 7 3 testcases/01.word 2 testcases/01.loc output/
# ./judge output/TEST01 7 testcases/01.ans
# srun -N2 -c12 ./hw4 TEST02 7 2 testcases/02.word 2 testcases/02.loc output/
# ./judge output/TEST02 7 testcases/02.ans
# srun -N3 -c10 ./hw4 TEST03 10 2 testcases/03.word 5 testcases/03.loc output/
# ./judge output/TEST03 10 testcases/03.ans
# srun -N3 -c12 ./hw4 TEST04 10 3 testcases/04.word 5 testcases/04.loc output/
# ./judge output/TEST04 10 testcases/04.ans
# srun -N3 -c12 ./hw4 TEST05 10 1 testcases/05.word 2 testcases/05.loc output/
# ./judge output/TEST05 10 testcases/05.ans
# srun -N4 -c10 ./hw4 TEST06 12 3 testcases/06.word 3 testcases/06.loc output/
# ./judge output/TEST06 12 testcases/06.ans
# srun -N4 -c12 ./hw4 TEST07 13 3 testcases/07.word 4 testcases/07.loc output/
# ./judge output/TEST07 13 testcases/07.ans
# srun -N4 -c12 ./hw4 TEST08 12 3 testcases/08.word 6 testcases/08.loc output/
# ./judge output/TEST08 12 testcases/08.ans
# srun -N4 -c12 ./hw4 TEST09 7 3 testcases/09.word 5 testcases/09.loc output/
# ./judge output/TEST09 7 testcases/09.ans
srun -N4 -c12 ./hw4 TEST10 12 5 testcases/10.word 10 testcases/10.loc output/
./judge output/TEST10 12 testcases/10.ans