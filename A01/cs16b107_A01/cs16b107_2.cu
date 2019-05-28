#include <stdio.h>

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	float bandwidth = ((prop.memoryBusWidth/8.0)*(prop.memoryClockRate*1000.0)*2/*ddr ram*/)/1000000000.0;
	printf("memory bus width (bits) = %d\n",prop.memoryBusWidth);
	printf("memory clock rate (kHz) = %d\n",prop.memoryClockRate);
	printf("memory bandwidth = %f\n",bandwidth);	
	
	return 0;
}