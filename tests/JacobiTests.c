#include "Jacobi.h"

void dluTest(){
	float *a = initMat2DHelper(false, 2);
	float *dinv = initMat2DHelper(false, 2);
	float *l = initMat2DHelper(false, 2);
	float *u = initMat2DHelper(false, 2);
	
	a[0] = 2.0;
	a[1] = 1.0;
	a[2] = 5.0;
	a[3] = 7.0;
	
	dluDecompHelper(a, dinv, l, u, 2);
	for(int i = 0; i < 4; i++){
		if(dinv[i] != 0){
			dinv[i] = 1 / dinv[i];
		}
		if(abs(a[i] - (dinv[i] + l[i] + u[i])) > 0.01){
			printf("LU decomposition unsuccessfull\n");
			exit(1);
		}
	}
}

void multiplyMats2D1DTest(){
	int testNum = 2;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** avals = (float**) malloc(sizeof(float*) * testNum);
	float** bvals = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	
	//Test 1
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float avals0[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	avals[0] = avals0;
	float bvals0[3] = {1, 2, 3};
	bvals[0] = bvals0;
	float answers0[3] = {14, 32, 50};
	answers[0] = answers0;

	//Test 2
	float params1[4] = {16, 4, 1, 4};
	params[1] = params1;
	float avals1[16 * 16] = {
						90, 9, 6, 4, 1, -5, 3, 9, 0, -3, 1, 5, 7, -9, 3, 2,
						3, 86, 4, 5, 7, 8, 3, 3, 4, 5, 6, 0, -3, 2, 3, 9,
						-5, 2, 89, 1, 4, 5, -2, -8, 7, 3, 4, 5, 2, 1, -1, 8,
						9, 9, 8, 69, -7, 2, 1, 8, 7, 2, 4, -9, 1, 1, 0, 0,
						3, 4, 1, 8, 95, 6, 5, 9, 8, 7, 6, 5, 4, -2, 3, 1,
						3, 2, 1, 8, 7, 87, 5, 2, 1, 4, 9, 8, 9, 2, 1, 0,
						3, 4, 1, 2, 3, 4, 91, 1, 9, 8, 7, 6, 5, 6, 7, 2,
						1, 1, 1, 3, 4, 2, 4, 92, 2, 1, -8, 2, 3, 2, 2, 1,
						0, 0, 0, 1, 1, 3, -2, 5, 93, -5, 4, 2, -3, 1, 0, 9,
						2, 1, 2, 1, 2, 3, 1, -8, 2, 95, 4, 8, 2, 1, 4, 0,
						2, 9, 8, 9, 8, 7, 2, 6, 4, 2, 89, 2, 1, 8, 7, 9,
						-5, -5, -4, -2, -1, 5, 4, 8, 7, 6, 5, 89, 0, 9, 3, 1,
						1, 1, 3, 2, 8, 8, 9, 7, 3, -3, 4, 2, 79, 9, 0, -4,
						2, 1, 3, -5, 8, 9, -2, 2, 1, 2, 3, -9, -3, 77, 1, 3,
						-2, 1, 7, 8, 9, 2, 0, 1, -1, -1, 2, -3, 2, 4, 92, -4,
						3, 2, -1, -1, -1, 0, 9, 8, 3, 4, 5, 3, 2, 1, 9, 90
						};
	avals[1] = avals1;
	float bvals1[16] = {9, -8, 2, 9, -3, 2, 8, 7, 2, 1, 0, -3, 4, 5, -9, 1};
	bvals[1] = bvals1;
	float answers1[16] = {800, -575, 84, 787, -113, 311, 746, 702, 208, 27, 85, -158, 492, 361, -754, 154};
	answers[1] = answers1;

	//Test 3
	float params2[4] = {16, 1, 2, 8};
	params[1] = params2;
	avals[2] = avals1;
	answers[2] = answers1;

	int lN;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		float* c = multiplyMats2D1DHelper(avals[i], bvals[i], lN);
		for(int j = 0; j < lN; j++){
			if(abs(c[j] - answers[i][j]) > .01){
				printf("multiplyMats2D1D test failed\n");
				printf("Currently is\n");
				printMat1DHelper(c, lN);
				printf("\n");
				printf("Should be\n");
				printMat1DHelper(answers[i], lN);
				exit(1);
			}
		}
	}
}

void multiplyMats2DTest(){
	int testNum = 1;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** avals = (float**) malloc(sizeof(float*) * testNum);
	float** bvals = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float avals0[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	avals[0] = avals0;
	float bvals0[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	bvals[0] = bvals0;
	float answer0[9] = {30, 36, 42, 66, 81, 96, 102, 126, 150};
	answers[0] = answer0;

	int lN;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		float*c = multiplyMats2DHelper(avals[i], bvals[i], lN);
		
		for(int j = 0; j < lN * lN; j++){
			if(c[j] != answers[i][j]){
				printf("multiplyMats2D test failed\n");
				printMat2DHelper(c, lN);
				exit(1);
			}
		}
	}
}

void jacobiMethodTest(){
	int testNum = 3;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** avals = (float**) malloc(sizeof(float*) * testNum);
	float** bvals = (float**) malloc(sizeof(float*) * testNum);
	float** xvals = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	
	//Test 1
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float avals0[9] = {4, -1, -1, -2, 6, 1, -1, 1, 7};
	avals[0] = avals0;
	float bvals0[3] = {3, 9, -6};
	bvals[0] = bvals0;
	float xvals0[3] = {0, 0, 0};
	xvals[0] = xvals0;
	float answer0[3] = {1, 2, -1};
	answers[0] = answer0;

	//Test 2
	float params1[4] = {16, 4, 1, 4};
	params[1] = params1;
	float avals1[16 * 16] = {
						90, 9, 6, 4, 1, -5, 3, 9, 0, -3, 1, 5, 7, -9, 3, 2,
						3, 86, 4, 5, 7, 8, 3, 3, 4, 5, 6, 0, -3, 2, 3, 9,
						-5, 2, 89, 1, 4, 5, -2, -8, 7, 3, 4, 5, 2, 1, -1, 8,
						9, 9, 8, 69, -7, 2, 1, 8, 7, 2, 4, -9, 1, 1, 0, 0,
						3, 4, 1, 8, 95, 6, 5, 9, 8, 7, 6, 5, 4, -2, 3, 1,
						3, 2, 1, 8, 7, 87, 5, 2, 1, 4, 9, 8, 9, 2, 1, 0,
						3, 4, 1, 2, 3, 4, 91, 1, 9, 8, 7, 6, 5, 6, 7, 2,
						1, 1, 1, 3, 4, 2, 4, 92, 2, 1, -8, 2, 3, 2, 2, 1,
						0, 0, 0, 1, 1, 3, -2, 5, 93, -5, 4, 2, -3, 1, 0, 9,
						2, 1, 2, 1, 2, 3, 1, -8, 2, 95, 4, 8, 2, 1, 4, 0,
						2, 9, 8, 9, 8, 7, 2, 6, 4, 2, 89, 2, 1, 8, 7, 9,
						-5, -5, -4, -2, -1, 5, 4, 8, 7, 6, 5, 89, 0, 9, 3, 1,
						1, 1, 3, 2, 8, 8, 9, 7, 3, -3, 4, 2, 79, 9, 0, -4,
						2, 1, 3, -5, 8, 9, -2, 2, 1, 2, 3, -9, -3, 77, 1, 3,
						-2, 1, 7, 8, 9, 2, 0, 1, -1, -1, 2, -3, 2, 4, 92, -4,
						3, 2, -1, -1, -1, 0, 9, 8, 3, 4, 5, 3, 2, 1, 9, 90
						};
	avals[1] = avals1;
	float bvals1[16] = {800, -575, 84, 787, -113, 311, 746, 702, 208, 27, 85, -158, 492, 361, -754, 154};
	bvals[1] = bvals1;
	float xvals1[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	xvals[1] = xvals1;
	float answers1[16] = {9, -8, 2, 9, -3, 2, 8, 7, 2, 1, 0, -3, 4, 5, -9, 1};
	answers[1] = answers1;

	//Test 3
	float params2[4] = {16, 1, 2, 8};
	params[2] = params2;
	avals[2] = avals1;
	bvals[2] = bvals1;
	xvals[2] = xvals1;
	answers[2] = answers1;

	int lN;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		clock_t begin = clock();
		float* x = jacobiMethodHelper(avals[i], bvals[i], xvals[i], lN);
		clock_t end = clock();
		
		for(int j = 0; j < lN; j++){
			if(abs(x[j] - answers[i][j]) > .01){
				printf("Jacobi Method test failed\n");
				printf("Currently is\n");
				printMat1DHelper(xvals[i], lN);
				printf("\n");
				printf("Should be\n");
				printMat1DHelper(answers[i], lN);
				exit(1);
			}
		}
		printf("Time elapsed is %f\n", (double) (end - begin) / CLOCKS_PER_SEC);
	}
}

int main(){
	printf("Running Tests\n");
	dluTest();
	multiplyMats2D1DTest();
	multiplyMats2DTest();
	jacobiMethodTest();
	printf("If you got here all tests passed\n");
	return 0;
}
