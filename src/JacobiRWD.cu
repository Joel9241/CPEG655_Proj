#include "JacobiRWD.cuh"

float* initMat2D(bool init, bool host){
	return initMat2DHelper(init, host, N);
}

float* initMat2DHelper(bool init, bool host, int size){
	float* m;
	size_t sizeMat = sizeof(float) * size * size;
	if(host){
		cudaMallocHost((void **) &m, sizeMat);
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
		cudaMallocHost((void **) &m, sizeMat);
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
	for(int i = 0; i < N; i++){
		printf("%f\n", mat[i]);
	}

}
void printMat2D(float* mat){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", mat[(i * N) + j]);
		}
		printf("\n");
	}
}

__global__ void multiplyMats2D(float* a, float* b, float* c){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < NB; i++){
		for(int j = 0; j < NB; j++){
			float sum = 0;
			for(int k = 0; k < N; k++){
				float tmp1 = a[(by * N * NT * NB) + (ty * N * NB) + (j * N) + k];
				float tmp2 = b[(bx * NT * NB) + (tx * NB) + (k * N) + i];
				sum += tmp1 * tmp2;
			}
			c[(by * N * NT * NB) + (bx * NT * NB) + (ty * N * NB) + (j * N) + i + (tx * NB)] = sum;
		}
	}
}

__global__ void multiplyMats2D1D(float* a, float* b, float* c){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < NB; i++){
		float sum = 0;
		for(int k = 0; k < N; k++){
			float tmp1 = a[(by * N * NT * NB) + (ty * N * NB) + k];
			float tmp2 = b[(bx * NT * NB) + (tx * NB) + (k * N) + i];
			sum += tmp1 * tmp2;
		}
		c[(by * N * NT * NB) + (bx * NT * NB) + (ty * N * NB) + i + (tx * NB)] = sum;
	}
}

void addMats2D(float* a, float* b, float* c){
	for(int i = 0; i < N * N; i++){
		c[i] = a[i] + b[i];
	}
}

void subMats2D(float* a, float* b, float* c){
	for(int i = 0; i < N * N; i++){
		c[i] = a[i] - b[i];
	}
}

void addMats1D(float* a, float* b, float* c){
	for(int i = 0; i < N; i++){
		c[i] = a[i] + b[i];
	}
}

void subMats1D(float* a, float* b, float* c){
	for(int i = 0; i < N; i++){
		c[i] = a[i] - b[i];
	}
}

float* jacobiMethod(float* a, float* b, float* x){
	/*
	Mat2D* dinv = initMat2D(false);
	Mat2D* l = initMat2D(false);
	Mat2D* u = initMat2D(false);
	dluDecomp(a, dinv, l, u);
	int i = 0;
	while(i < 25){
		x = jacobiIterate(dinv, l, u, b, x);
		i++;
	}
	*/
	return x;
}

float* jacobiIterate(float* dinv, float* l, float* u, float* b, float* x){
	/*
	Mat2D* lu = addMats2D(l, u);
	Mat1D* lux = multiplyMats2D1D(lu, x);
	Mat1D* blux = subMats1D(b, lux);
	x = multiplyMats2D1D(dinv, blux);
	*/
	return x;
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
