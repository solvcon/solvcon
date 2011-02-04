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

extern "C" void structop(int nthread, custom *ctm, void *gp) {
	int nblock = (ctm->nelm + nthread - 1) / nthread;
    cuda_structop<<<nblock, nthread>>>((custom *)gp);
    cudaThreadSynchronize();
};
