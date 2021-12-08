#include "Jacobi.h"

Mat2D* initMat2D(bool init){
	Mat2D* m = (Mat2D*) malloc(sizeof(Mat2D*));
	m->size = N;
	m->mat = (float*) malloc(sizeof(float) * N * N);
	if(!init){
		return m;
	}
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			m->mat[(i * N) + j] = 5;
		}
	}
	return m;
}

Mat1D* initMat1D(bool init){
	Mat1D* m = (Mat1D*) malloc(sizeof(Mat1D*));
	m->size = N;
	m->mat = (float*) malloc(sizeof(float) * N);
	if(!init){
		return m;
	}
	for(int i = 0; i < N; i++){
		m->mat[i] = 5;
	}
	return m;
}

void printMat1D(Mat1D* mat){
	for(int i = 0; i < N; i++){
		printf("%f\n", mat->mat[i]);
	}

}
void printMat2D(Mat2D* mat){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", mat->mat[(i * N) + j]);
		}
		printf("\n");
	}
}

__global__ void multiplyMats2D(Mat2D* a, Mat2D* b, Mat2D* c){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < NB; i++){
		for(int j = 0; j < NB; j++){
			float sum = 0;
			for(int k = 0; k < N; k++){
				float tmp1 = a->mat[(by * N * NT * NB) + (ty * N * NB) + (j * N) + k];
				float tmp2 = b->mat[(bx * NT * NB) + (tx * NB) + (k * N) + i];
				sum += tmp1 * tmp2;
			}
			c->mat[(by * N * NT * NB) + (bx * NT * NB) + (ty * N * NB) + (j * N) + i + (tx * NB)] = sum;
		}
	}
}

__global__ void multiplyMats2D1D(Mat2D* a, Mat1D* b, Mat1D* c){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	for(int i = 0; i < NB; i++){
		float sum = 0;
		for(int k = 0; k < N; k++){
			float tmp1 = a->mat[(by * N * NT * NB) + (ty * N * NB) + k];
			float tmp2 = b->mat[(bx * NT * NB) + (tx * NB) + (k * N) + i];
			sum += tmp1 * tmp2;
		}
		c->mat[(by * N * NT * NB) + (bx * NT * NB) + (ty * N * NB) + i + (tx * NB)] = sum;
	}
}

void addMats2D(Mat2D* a, Mat2D* b, Mat2D* c){
	for(int i = 0; i < N * N; i++){
		c->mat[i] = a->mat[i] + b->mat[i];
	}
}

void subMats2D(Mat2D* a, Mat2D* b, Mat2D* c){
	for(int i = 0; i < N * N; i++){
		c->mat[i] = a->mat[i] - b->mat[i];
	}
}

void addMats1D(Mat1D* a, Mat1D* b, Mat1D* c){
	for(int i = 0; i < N; i++){
		c->mat[i] = a->mat[i] + b->mat[i];
	}
}

void subMats1D(Mat1D* a, Mat1D* b, Mat1D* c){
	for(int i = 0; i < N; i++){
		c->mat[i] = a->mat[i] - b->mat[i];
	}
}

Mat1D* jacobiMethod(Mat2D* a, Mat1D* b, Mat1D* x){
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

Mat1D* jacobiIterate(Mat2D* dinv, Mat2D* l, Mat2D* u, Mat1D* b, Mat1D* x){
	/*
	Mat2D* lu = addMats2D(l, u);
	Mat1D* lux = multiplyMats2D1D(lu, x);
	Mat1D* blux = subMats1D(b, lux);
	x = multiplyMats2D1D(dinv, blux);
	*/
	return x;
}

void dluDecomp(Mat2D* a, Mat2D* dinv, Mat2D* l, Mat2D* u){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if(i == j){
				dinv->mat[(i * N) + j] = 1 / a->mat[(i * N) + j];
			}
			else if(i > j){
				dinv->mat[(i * N) + j] = 0;
				l->mat[(i * N) + j] = a->mat[(i * N) + j];
			}
			else{
				dinv->mat[(i * N) + j] = 0;
				u->mat[(i * N) + j] = a->mat[(i * N) + j];
			}
		}
	}
}
