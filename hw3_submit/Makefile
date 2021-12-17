NVFLAGS3  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
NVFLAGS2  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
CXXFLAGS := -O3 -pthread -fopenmp
LDFLAGS  := -lm
EXES     := hw3-3 hw3-2 hw3-1

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS2) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc  $(NVFLAGS3) $(LDFLAGS) -o $@ $?




