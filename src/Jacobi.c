#include "Jacobi.h"

float* initX(){
	return calloc(0, sizeof(float) * N);
}

float* initMat2D(bool init){
	return initMat2DHelper(init, N);
}

float* initMat2DHelper(bool init, int lN){
	float* m = malloc(sizeof(float) * lN * lN);
	if(!init){
		return m;
	}
	for(int i = 0; i < lN; i++){
		int sum = 0;
		int randNum = (rand() % 23) - 10;
		for(int j = 0; j < lN; j++){
			if(i != j){
				m[(i * lN) + j] = randNum;
				sum += randNum;
			}
		}
		m[(i * lN) + i] = (rand() % 23) - 10 + sum;
	}
	return m;
}

float* initMat1D(bool init){
	return initMat1DHelper(init, N);
}

float* initMat1DHelper(bool init, int lN){
	float* m = malloc(sizeof(float) * lN);
	if(!init){
		return m;
	}
	for(int i = 0; i < lN; i++){
		m[i] = 5;
	}
	return m;
}

void printMat1D(float* mat){
	printMat1DHelper(mat, N);
}

void printMat1DHelper(float* mat, int lN){
	for(int i = 0; i < lN; i++){
		printf("%f\n", mat[i]);
	}
}

void printMat2D(float* mat){
	printMat2DHelper(mat, N);
}

void printMat2DHelper(float* mat, int lN){
	for(int i = 0; i < lN; i++){
		for(int j = 0; j < lN; j++){
			printf("%f ", mat[(i * lN) + j]);
		}
		printf("\n");
	}
}

float* multiplyMats2D(float* a, float* b){
	return multiplyMats2DHelper(a, b, N);
}

float* multiplyMats2DHelper(float* a, float* b, int lN){
	float* c = initMat2DHelper(false, lN);
	for(int i = 0; i < lN; i++){
		for(int j = 0; j < lN; j++){
			float sum = 0;
			for(int k = 0; k < lN; k++){
				float tmp1 = a[(i * lN) + k];
				float tmp2 = b[(k * lN) + j];
				sum += tmp1 * tmp2;
			}
			c[(i * lN) + j] = sum;
		}
	}
	return c;
}

float* multiplyMats2D1D(float* a, float* b){
	return multiplyMats2D1DHelper(a, b, N);
}

float* multiplyMats2D1DHelper(float* a, float* b, int lN){
	float* c = initMat1DHelper(false, lN);
	for(int i = 0; i < lN; i++){
		float sum = 0;
		for(int k = 0; k < lN; k++){
			float tmp1 = a[(i * lN) + k];
			float tmp2 = b[k];
			sum += tmp1 * tmp2;
		}
		c[i] = sum;
	}
	return c;
}

float* addMats2D(float* a, float* b){
	return addMats2DHelper(a, b, N);
}

float* addMats2DHelper(float* a, float* b, int lN){
	float* c = initMat2DHelper(false, lN);
	for(int i = 0; i < lN * lN; i++){
		c[i] = a[i] + b[i];
	}
	return c;
}

float* subMats2D(float* a, float* b){
	return subMats2DHelper(a, b, N);
}

float* subMats2DHelper(float* a, float* b, int lN){
	float* c = initMat2DHelper(false, lN);
	for(int i = 0; i < lN * lN; i++){
		c[i] = a[i] - b[i];
	}
	return c;
}

float* addMats1D(float* a, float* b){
	return addMats1DHelper(a, b, N);
}

float* addMats1DHelper(float* a, float* b, int lN){
	float* c = initMat1DHelper(false, lN);
	for(int i = 0; i < lN; i++){
		c[i] = a[i] + b[i];
	}
	return c;
}

float* subMats1D(float* a, float* b){
	return subMats1DHelper(a, b, N);
}

float* subMats1DHelper(float* a, float* b, int lN){
	float* c = initMat1DHelper(false, lN);
	for(int i = 0; i < lN; i++){
		c[i] = a[i] - b[i];
	}
	return c;
}

float* jacobiMethod(float* a, float* b, float* x){
	return jacobiMethodHelper(a, b, x, N);
}

float* jacobiMethodHelper(float* a, float* b, float* x, int lN){
	float* dinv = initMat2DHelper(false, lN);
	float* l = initMat2DHelper(false, lN);
	float* u = initMat2DHelper(false, lN);
	dluDecompHelper(a, dinv, l, u, lN);
	int i = 0;
	while(i < 25){
		x = jacobiIterateHelper(dinv, l, u, b, x, lN);
		i++;
	}
	return x;
}

float* jacobiIterate(float* dinv, float* l, float* u, float* b, float* x){
	return jacobiIterateHelper(dinv, l, u, b, x, N);
}

float* jacobiIterateHelper(float* dinv, float* l, float* u, float* b, float* x, int lN){
	float* lu = addMats2DHelper(l, u, lN);
	float* lux = multiplyMats2D1DHelper(lu, x, lN);
	float* blux = subMats1DHelper(b, lux, lN);
	x = multiplyMats2D1DHelper(dinv, blux, lN);
	return x;
}

void dluDecomp(float* a, float* dinv, float* l, float* u){
	dluDecompHelper(a, dinv, l, u, N);
}

void dluDecompHelper(float* a, float* dinv, float* l, float* u, int lN){
	for(int i = 0; i < lN; i++){
		for(int j = 0; j < lN; j++){
			if(i == j){
				dinv[(i * lN) + j] = 1 / a[(i * lN) + j];
			}
			else if(i > j){
				dinv[(i * lN) + j] = 0;
				l[(i * lN) + j] = a[(i * lN) + j];
			}
			else{
				dinv[(i * lN) + j] = 0;
				u[(i * lN) + j] = a[(i * lN) + j];
			}
		}
	}
}
