typedef struct {
    int nelm;
    double dval;
    double* arra;
    double* arrb;
    double* arrc;
} custom;

__global__ void cuda_structop(custom *ctm) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ctm->nelm)
        ctm->arrc[i] = ctm->arra[i] + ctm->arrb[i] + ctm->dval;
};

extern "C" void structop(custom *ctm, void *gp) {
    int threadsPerBlock = 64;
	int blocksPerGrid = ctm->nelm / threadsPerBlock;
    cuda_structop<<<blocksPerGrid, threadsPerBlock>>>((custom *)gp);
    cudaThreadSynchronize();
};
