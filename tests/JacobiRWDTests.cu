#include "JacobiRWD.cuh"

void dluTest(){
	float *a = initMat2D(false, true);
	float *dinv = initMat2D(false, true);
	float *l = initMat2D(false, true);
	float *u = initMat2D(false, true);
	
	a[0] = 2;
	a[1] = 1;
	a[2] = 5;
	a[3] = 7;
	
	dluDecomp(a, dinv, l, u);
	for(int i = 0; i < N; i++){
		if(dinv[i] != 0){
			dinv[i] = 1 / dinv[i];
		}
		if(a[i] != dinv[i] + l[i] + u[i]){
			printf("LU decomposition unsuccessfull\n");
			exit(1);
		}
	}
}

void multiplyMats2D1DTest(){
	int testNum = 1;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float answers0[3] = {14, 32, 50};
	answers[0] = answers0;

	int lN;
	int lNT;
	int lNB;
	int lNK;
	float* h_a;
	float* h_b;
	float* h_c;
	float* d_a;
	float* d_b;
	float* d_c;
	size_t size1;
	size_t size2;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		lNT = params[i][1];
		lNB = params[i][2];
		lNK = params[i][3];
		dim3 threadPerBlock(lNT, 1);
		dim3 blockPerGrid(lNK, lNK);
		
		h_a = initMat2DHelper(false, true, lN);
		h_b = initMat1DHelper(false, true, lN);
		h_c = initMat1DHelper(false, true, lN);
		
		d_a = initMat2DHelper(false, false, lN);
		d_b = initMat1DHelper(false, false, lN);
		d_c = initMat1DHelper(false, false, lN);	

		for(int j = 0; j < lN * lN; j++){
			h_a[j] = j + 1;
		}
		for(int j = 0; j < lN; j++){
			h_b[j] = j + 1;
		}

		size1 = lN * lN * sizeof(float);
		size2 = lN * sizeof(float);

		cudaMemcpy(d_a, h_a, size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, size2, cudaMemcpyHostToDevice);

		multiplyMats2D1DTB<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c, lN, lNT, lNB);
		cudaMemcpy(h_c, d_c, size2, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		
		for(int j = 0; j < lN; j++){
			if(h_c[j] != answers[i][j]){
				printf("multiplyMats2D1D test failed\n");
				printMat1DHelper(h_c, lN);
				exit(1);
			}
		}
	}
}

void multiplyMats2DTest(){
	int testNum = 1;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float answer0[9] = {30, 36, 42, 66, 81, 96, 102, 126, 150};
	answers[0] = answer0;

	int lN;
	int lNT;
	int lNB;
	int lNK;
	float* h_a;
	float* h_b;
	float* h_c;
	float* d_a;
	float* d_b;
	float* d_c;
	size_t size;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		lNT = params[i][1];
		lNB = params[i][2];
		lNK = params[i][3];
		dim3 threadPerBlock(lNT, lNT);
		dim3 blockPerGrid(lNK, lNK);
		
		h_a = initMat2DHelper(false, true, lN);
		h_b = initMat2DHelper(false, true, lN);
		h_c = initMat2DHelper(false, true, lN);
		
		d_a = initMat2DHelper(false, false, lN);
		d_b = initMat2DHelper(false, false, lN);
		d_c = initMat2DHelper(false, false, lN);	

		for(int j = 0; j < lN * lN; j++){
			h_a[j] = j + 1;
			h_b[j] = j + 1;
		}

		size = lN * lN * sizeof(float);

		cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

		multiplyMats2DTB<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c, lN, lNT, lNB);
		cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		
		for(int j = 0; j < lN * lN; j++){
			if(h_c[j] != answers[i][j]){
				printf("multiplyMats2D test failed\n");
				printMat2DHelper(h_c, lN);
				exit(1);
			}
		}
	}
}

void jacobiMethodTest(){
	/*
	float* h_a = initMat2D(false, true, 2);
	float* h_b = initMat1D(false, true, 2);
	float* h_x = initMat1D(false, true, 2);

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

	int testNum = 1;
	float** params = (float**) malloc(sizeof(float*) * testNum);
	float** answers = (float**) malloc(sizeof(float*) * testNum);
	float params0[4] = {3, 3, 1, 1};
	params[0] = params0;
	float answer0[9] = {30, 36, 42, 66, 81, 96, 102, 126, 150};
	answers[0] = answer0;

	int lN;
	int lNT;
	int lNB;
	int lNK;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		lNT = params[i][1];
		lNB = params[i][2];
		lNK = params[i][3];
		dim3 threadPerBlock(lNT, lNT);
		dim3 blockPerGrid(lNK, lNK);
		
		float* h_a = initMat2DHelper(false, true, lN);
		float* h_b = initMat1DHelper(false, true, lN);
		float* h_x = initMat1DHelper(false, true, lN);
		float* h_dinv = initMat2DHelper(false, true, lN);
		float* h_l = initMat2DHelper(false, true, lN);
		float* h_u = initMat2DHelper(false, true, lN);
		
		float* d_a = initMat2DHelper(false, false, lN);
		float* d_b = initMat1DHelper(false, false, lN);
		float* d_x = initMat1DHelper(false, false, lN);
		float* d_dinv = initMat2DHelper(false, false, lN);
		float* d_l = initMat2DHelper(false, false, lN);
		float* d_u = initMat2DHelper(false, false, lN);

		for(int j = 0; j < lN * lN; j++){
			h_a[j] = j + 1;
			h_b[j] = j + 1;
		}

		size_t size1 = lN * lN * sizeof(float);
		size_t size2 = lN * sizeof(float);

		cudaMemcpy(d_a, h_a, size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, size2, cudaMemcpyHostToDevice);
		cudaMemcpy(d_x, h_x, size2, cudaMemcpyHostToDevice);
		cudaMemcpy(d_dinv, h_dinv, size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_l, h_l, size1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_u, h_u, size1, cudaMemcpyHostToDevice);

		jacobiMethodTB<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_x, d_dinv, d_l, d_u, lN, lNT, lNB);
		cudaMemcpy(h_x, d_x, size2, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		
		for(int j = 0; j < lN * lN; j++){
			if(h_x[j] != answers[i][j]){
				printf("Jacobi Method test failed\n");
				printMat1DHelper(h_x, lN);
				exit(1);
			}
		}
	}
}

int main(){
	printf("Running Tests\n");
	dluTest();
	multiplyMats2D1DTest();
	multiplyMats2DTest();
	/*
	jacobiMethodTest();
	*/
	printf("If you got here all tests passed\n");
	return 0;
}
