__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
};

extern "C" void invoke_VecAdd(float* d_A, float* d_B, float* d_C, int N) {
    int threadsPerBlock = 256;
	int blocksPerGrid = N / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
};
