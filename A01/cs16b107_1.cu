#include <stdio.h>

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("%d\n",prop.localL1CacheSupported);
	printf("%d\n",prop.globalL1CacheSupported);
	printf("%d\n",prop.l2CacheSize);
	printf("%d\n",prop.maxThreadsPerBlock);
	printf("%d\n",prop.regsPerBlock);
	printf("%d\n",prop.regsPerMultiprocessor);
	printf("%d\n",prop.warpSize);
	printf("%lu\n",prop.totalGlobalMem);	
	
	return 0;
}