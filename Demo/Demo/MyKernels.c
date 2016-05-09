// note: the .cu suffix would be more typical, but the .c suffix
// makes it easier to setup in VS without requiring any configuration
// changes

typedef struct
{
	int Id;
	unsigned int Value;
} SomeBasicType;

extern "C"
{
	// the __global__ here makes it accessible as a callable kernel
	__global__ void Multiply(const int N, SomeBasicType* __restrict data, int factor)
	{
		// grid-stride loop
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
		{
			// note we want to mutate in place; we don't want to copy it out, update, copy back
			(data + i)->Value *= factor;
		}
	}
}
