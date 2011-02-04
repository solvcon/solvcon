extern "C" __global__
void cuda_vecadd_float(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
};

extern "C" void vecadd_float(float* d_A, float* d_B, float* d_C, int N) {
    int threadsPerBlock = 64;
	int blocksPerGrid = N / threadsPerBlock;
    cuda_vecadd_float<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
};
