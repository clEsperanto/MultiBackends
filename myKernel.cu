extern "C" __global__ void myKernel(int *dA, int *dB, int *dC, int n) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y; 

    if (i < n && j < n)
    {
        int c = 0;
        for (int k = 0; k < n; k++)
        {
            c += dA[i * n + k] * dB[k + j * n];
        }
        dC[i * n + j] = c;

        // printf("%d", n);
    }
}