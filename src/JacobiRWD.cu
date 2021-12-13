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
	/*
	int i = 0;
	while(i < 25){
		float* lu = initMat2DHelper(false, false, N);
		addMatsHelper(l, u, lu, N * N, NT, NB);
		float* lux = initMat1DHelper(false, false, N);
		multiplyMats2D1DHelper(lu, x, lux, N, NT, NB);
		float* blux = initMat1DHelper(false, false, N);
		subMatsHelper(b, lux, blux, N, NT, NB);
		multiplyMats2D1DHelper(dinv, blux, x, N, NT, NB);
		cudaDeviceSynchronize();
		i++;
	}
	*/
}

__host__ void jacobiMethodTB(float* h_a, float* h_b, float* h_x, int lN, int lNT, int lNB, int lNK){
	dim3 threadPerBlock(lNT, lNT);
	dim3 blockPerGrid(lNK, lNK);

	float* h_dinv = initMat2DHelper(false, true, lN);
	float* h_l = initMat2DHelper(false, true, lN);
	float* h_u = initMat2DHelper(false, true, lN);
	float* h_lu = initMat2DHelper(false, true, lN);
	float* h_lux = initMat1DHelper(false, true, lN);
	float* h_blux = initMat1DHelper(false, true, lN);
																 
	float* d_a = initMat2DHelper(false, false, lN);
	float* d_b = initMat1DHelper(false, false, lN);
	float* d_x = initMat1DHelper(false, false, lN);
	float* d_dinv = initMat2DHelper(false, false, lN);
	float* d_l = initMat2DHelper(false, false, lN);
	float* d_u = initMat2DHelper(false, false, lN);
	float* d_lu = initMat2DHelper(false, true, lN);
	float* d_lux = initMat1DHelper(false, true, lN);
	float* d_blux = initMat1DHelper(false, true, lN);
	
	dluDecompTB(h_a, h_dinv, h_l, h_u, lN, lNT, lNB);

	size_t size1 = lN * lN * sizeof(float);
	size_t size2 = lN * sizeof(float);

	cudaMemcpy(d_a, h_a, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dinv, h_dinv, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_l, h_l, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, h_u, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lu, h_lu, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lux, h_lux, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_blux, h_blux, size2, cudaMemcpyHostToDevice);

	int i = 0;
	while(i < 25){
		//printf("hello world\n");
		addMats2DTB<<<blockPerGrid, threadPerBlock>>>(d_l, d_u, d_lu, lN, lNT, lNB);
		cudaDeviceSynchronize();
		multiplyMats2D1DTB<<<blockPerGrid, threadPerBlock>>>(d_lu, d_x, d_lux, lN, lNT, lNB);
		cudaDeviceSynchronize();
		subMats1DTB<<<blockPerGrid, threadPerBlock>>>(d_b, d_lux, d_blux, lN, lNT, lNB);
		cudaDeviceSynchronize();
		multiplyMats2D1DTB<<<blockPerGrid, threadPerBlock>>>(d_dinv, d_blux, d_x, lN, lNT, lNB);
		cudaDeviceSynchronize();
		i++;
	}
	cudaMemcpy(h_x, d_x, size2, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
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

void dluDecompTB(float* a, float* dinv, float* l, float* u, int lN, int lNT, int lNB){
	for(int i = 0; i < lN; i++){
		for(int j = 0; j < lN; j++){
			if(i == j){
				dinv[(i * lN) + j] = 1 / a[(i * lN) + j];
				l[(i * lN) + j] = 0.0;
				u[(i * lN) + j] = 0.0;
			}
			else if(i > j){
				dinv[(i * lN) + j] = 0.0;
				l[(i * lN) + j] = a[(i * lN) + j];
				u[(i * lN) + j] = 0.0;
			}
			else{
				dinv[(i * lN) + j] = 0.0;
				l[(i * lN) + j] = 0.0;
				u[(i * lN) + j] = a[(i * lN) + j];
			}
		}
	}
}
