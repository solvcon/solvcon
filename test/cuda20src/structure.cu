typedef struct {
    int nelm;
    double dval;
    double* arr;
} custom;

__global__ void cuda_structop(custom *ctm) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ctm->nelm)
        //garr[i] += ctm->dval - ctm->fval + i;
        ctm->arr[i] = ctm->dval + i;
};

extern "C" void structop(custom *ctm, void *gp) {
    int threadsPerBlock = 64;
	int blocksPerGrid = ctm->nelm / threadsPerBlock;
    cuda_structop<<<blocksPerGrid, threadsPerBlock>>>((custom *)gp);
    cudaThreadSynchronize();
};
