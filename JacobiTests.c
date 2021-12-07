#include "Jacobi.h"

void dluTest(){
	Mat2D *a = initMat2D(false);
	Mat2D *dinv = initMat2D(false);
	Mat2D *l = initMat2D(false);
	Mat2D *u = initMat2D(false);
	
	a->mat[0] = 2;
	a->mat[1] = 1;
	a->mat[2] = 5;
	a->mat[3] = 7;
	
	dluDecomp(a, dinv, l, u);
	Mat2D* acopy = multiplyMats2D(l, u);
	for(int i = 0; i < N; i++){
		if(a->mat[i] != acopy->mat[i]){
			printf("LU decomposition unsuccessfull\n");
			exit(1);
		}
	}

}

int main(){
	printf("Running Tests\n");
	dluTest();
	printf("If you got here all tests passed\n");
	return 0;
}
