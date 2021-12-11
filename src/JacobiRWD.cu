#include "JacobiRWD.cuh"

float* initMat2D(bool init, bool host){
	return initMat2DHelper(init, host, N);
}

float* initMat2DHelper(bool init, bool host, int size){
	float* m;
	size_t sizeMat = sizeof(float) * size * size;
	if(host){
		m = (float*) malloc(sizeMat);
	}
	else{
		cudaMalloc((void **) &m, sizeMat);
	}
	if(!init){
		return m;
	}
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			m[(i * size) + j] = 5;
		}
	}
	return m;
}

float* initMat1D(bool init, bool host){
	return initMat1DHelper(init, host, N);
}

float* initMat1DHelper(bool init, bool host, int size){
	float* m;
	size_t sizeMat = sizeof(float) * size;
	if(host){
		m = (float*) malloc(sizeMat);
	}
	else{
		cudaMalloc((void **) &m, sizeMat);
	}
	if(!init){
		return m;
	}
	for(int i = 0; i < size; i++){
		m[i] = 5;
	}
	return m;
}

void printMat1D(float* mat){
	printMat1DHelper(mat, N);
}

void printMat1DHelper(float* mat, int size){
	for(int i = 0; i < size; i++){
		printf("%f\n", mat[i]);
	}
}

void printMat2D(float* mat){
	printMat2DHelper(mat, N);
}

void printMat2DHelper(float* mat, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("%f ", mat[(i * size) + j]);
		}
		printf("\n");
	}
}

__global__ void multiplyMats2D(float* a, float* b, float* c){
	multiplyMats2DHelper(a, b, c, N, NT, NB);
}

__global__ void multiplyMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	multiplyMats2DHelper(a, b, c, lN, lNT, lNB);
}

__device__ void multiplyMats2DHelper(float* a, float* b, float* c, int lN, int lNT, int lNB){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < lNB; i++){
		for(int j = 0; j < lNB; j++){
			float sum = 0;
			for(int k = 0; k < lN; k++){
				float tmp1 = a[(by * lN * lNT * lNB) + (ty * lN * lNB) + (j * lN) + k];
				float tmp2 = b[(bx * lNT * lNB) + (tx * lNB) + (k * lN) + i];
				sum += tmp1 * tmp2;
			}
			c[(by * lN * lNT * lNB) + (bx * lNT * lNB) + (ty * lN * lNB) + (j * lN) + i + (tx * lNB)] = sum;
		}
	}
}
__global__ void multiplyMats2D1D(float* a, float* b, float* c){
	multiplyMats2D1DHelper(a, b, c, N, NT, NB);
}

__global__ void multiplyMats2D1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	multiplyMats2D1DHelper(a, b, c, lN, lNT, lNB);
}

__device__ void multiplyMats2D1DHelper(float* a, float* b, float* c, int lN, int lNT, int lNB){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < lNB; i++){
		float sum = 0;
		for(int k = 0; k < lN; k++){
			float tmp1 = a[(by * lN * lNT * lNB) + (tx * lN * lNB) + k];
			float tmp2 = b[(by * lN * lNT * lNB) + (ty * lN * lNB) + k];
			sum += tmp1 * tmp2;
		}
		c[(by * lN * lNT * lNB) + (bx * lNT * lNB) + (ty * lN * lNB) + i + (tx * lNB)] = sum;
	}
}

__global__ void addMats2D(float* a, float* b, float* c){
	addMatsHelper(a, b, c, N * N, NT, NB);
}

__global__ void addMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	addMatsHelper(a, b, c, lN * lN, lNT, lNB);
}

__global__ void addMats1D(float* a, float* b, float* c){
	addMatsHelper(a, b, c, N, NT, NB);
}

__global__ void addMats1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	addMatsHelper(a, b, c, lN, lNT, lNB);
}

__device__ void addMatsHelper(float* a, float* b, float* c, int lN, int lNT, int lNB){
	for(int i = 0; i < lN; i++){
		c[i] = a[i] + b[i];
	}
}

__global__ void subMats2D(float* a, float* b, float* c){
	subMatsHelper(a, b, c, N * N, NT, NB);
}

__global__ void subMats2DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	subMatsHelper(a, b, c, lN * lN, lNT, lNB);
}

__global__ void subMats1D(float* a, float* b, float* c){
	subMatsHelper(a, b, c, N, NT, NB);
}

__global__ void subMats1DTB(float* a, float* b, float* c, int lN, int lNT, int lNB){
	subMatsHelper(a, b, c, lN, lNT, lNB);
}

__device__ void subMatsHelper(float* a, float* b, float* c, int lN, int lNT, int lNB){
	for(int i = 0; i < lN; i++){
		c[i] = a[i] - b[i];
	}
}

__global__ void jacobiMethod(float* a, float* b, float* x, float* dinv, float* l, float* u){
	jacobiMethodHelper(a, b, x, dinv, l, u, N, NT, NB);
}

__global__ void jacobiMethodTB(float* a, float* b, float* x, float* dinv, float* l, float* u, int lN, int lNT, int lNB){
	jacobiMethodHelper(a, b, x, dinv, l, u, lN, lNT, lNB);
}

__device__ void jacobiMethodHelper(float* a, float* b, float* x, float* dinv, float* l, float* u, int lN, int lNT, int lNB){
	int i = 0;
	while(i < 25){
		jacobiIterate(dinv, l, u, b, x, lN, lNT, lNB);
		cudaDeviceSynchronize();
		i++;
	}
}

__device__ void jacobiIterate(float* dinv, float* l, float* u, float* b, float* x, int lN, int lNT, int lNB){
	float* lu = initMat2DHelper(false, false, lN);
	addMats2DTB(l, u, lu, lN, lNT, lNB);
	float* lux = initMat1DHelper(false, false, lN);
	multiplyMats2D1DHelper(lu, x, lux, lN, lNT, lNB);
	float* blux = initMat1DHelper(false, false, lN);
	subMatsHelper(b, lux, blux, lN, lNT, lNB);
	multiplyMats2D1DHelper(dinv, blux, x, lN, lNT, lNB);
}

void dluDecomp(float* a, float* dinv, float* l, float* u){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if(i == j){
				dinv[(i * N) + j] = 1 / a[(i * N) + j];
			}
			else if(i > j){
				dinv[(i * N) + j] = 0;
				l[(i * N) + j] = a[(i * N) + j];
			}
			else{
				dinv[(i * N) + j] = 0;
				u[(i * N) + j] = a[(i * N) + j];
			}
		}
	}
}
