#include "Jacobi.h"

void dluTest(){
	/*
	Mat2D *a = initMat2D(false);
	Mat2D *dinv = initMat2D(false);
	Mat2D *l = initMat2D(false);
	Mat2D *u = initMat2D(false);
	
	a->mat[0] = 2;
	a->mat[1] = 1;
	a->mat[2] = 5;
	a->mat[3] = 7;
	
	dluDecomp(a, dinv, l, u);
	for(int i = 0; i < N; i++){
		if(dinv->mat[i] != 0){
			dinv->mat[i] = 1 / dinv->mat[i];
		}
		if(a->mat[i] != dinv->mat[i] + l->mat[i] + u->mat[i]){
			printf("LU decomposition unsuccessfull\n");
			exit(1);
		}
	}
	*/
}

void multiplyMats2D1DTest(){
	/*
	Mat2D *a = initMat2D(false);
	Mat1D *b = initMat1D(false);
	
	a->mat[0] = 1;
	a->mat[1] = 2;
	a->mat[2] = 3;
	a->mat[3] = 4;
	b->mat[0] = 2;
	b->mat[1] = 4;

	Mat1D *c = multiplyMats2D1D(a, b);
	if((c->mat[0] != 10) || (c->mat[1] != 22)){
		printf("multiplyMats2D1D unsuccessfull\n");
		exit(1);
	}
	*/
}

void jacobiMethodTest(){
	/*
	Mat2D *a = initMat2D(false);
	Mat1D *b = initMat1D(false);
	Mat1D *x = initMat1D(false);

	a->mat[0] = 2;
	a->mat[1] = 1;
	a->mat[2] = 5;
	a->mat[3] = 7;
	b->mat[0] = 11;
	b->mat[1] = 13;
	x->mat[0] = 1;
	x->mat[1] = 1;
	x = jacobiMethod(a, b, x);
	*/
}

int main(){
	printf("Running Tests\n");
	dluTest();
	multiplyMats2D1DTest();
	jacobiMethodTest();
	printf("If you got here all tests passed\n");
	return 0;
}
