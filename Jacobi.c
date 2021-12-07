#include "Jacobi.h"

Mat2D* initMat2D(bool init){
	Mat2D* m = malloc(sizeof(Mat2D*));
	m->size = N;
	m->mat = malloc(sizeof(float) * N * N);
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
	Mat1D* m = malloc(sizeof(Mat1D*));
	m->size = N;
	m->mat = malloc(sizeof(float) * N);
	if(!init){
		return m;
	}
	for(int i = 0; i < N; i++){
		m->mat[i] = 5;
	}
	return m;
}

void printMat2D(Mat2D* mat){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", mat->mat[(i * N) + j]);
		}
		printf("\n");
	}
}

Mat2D* multiplyMats2D(Mat2D* a, Mat2D* b){
	Mat2D* c = initMat2D(false);
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			float sum = 0;
			for(int k = 0; k < N; k++){
				float tmp1 = a->mat[(i * N) + k];
				float tmp2 = b->mat[(k * N) + j];
				sum += tmp1 * tmp2;
			}
			c->mat[(i * N) + j] = sum;
		}
	}
	return c;
}

void dluDecomp(Mat2D* a, Mat2D* dinv, Mat2D* l, Mat2D* u){
	//calc dinv
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if(i == j){
				dinv->mat[(i * N) + j] = 1 / a->mat[(i * N) + j];
			}
			else{
				dinv->mat[(i * N) + j] = 0;
			}
		}
	}
	for(int i = 0; i < N; i++){
		//calc l
		for(int j = 0; j < N; j++){
			if(j < i){
				l->mat[(j * N) + i] = 0;
			}
			else{
				l->mat[(j * N) + i] = a->mat[(j * N) + i];
				for(int k = 0; k < i; k++){
					l->mat[(j * N) + i] = l->mat[(j * N) + i] - l->mat[(j * N) + k] * u->mat[(k * N) + i];
				}
			}
		}
		//calc u
		for(int j = 0; j < N; j++){
			if(j < i){
				u->mat[(i * N) + j] = 0;
			}
			else if(j == i){
				u->mat[(i * N) + j] = 1;
			}
			else{
				u->mat[(i * N) + j] = a->mat[(i * N) + j] / l->mat[(i * N) + i];
				for(int k = 0; k < i; k++){
					u->mat[(i * N) + j] = u->mat[(i * N) + j] - ((l->mat[(i * N) + k] * u->mat[(k * N) + j]) / l->mat[(i * N) + i]);
				}
			}
		}
	}
}
