#pragma once

void ChangeHeapSize(size_t heapSize) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("Heap Size=%ld\n", heapSize);

}

void ChangeStackSize(size_t stackSize){
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack Size=%ld\n", stackSize);
}