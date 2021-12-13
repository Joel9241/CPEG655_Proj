#include "JacobiRWD.cuh"

void dluTest(){
	float *a = initMat2DHelper(false, true, 2);
	float *dinv = initMat2DHelper(false, true, 2);
	float *l = initMat2DHelper(false, true, 2);
	float *u = initMat2DHelper(false, true, 2);
	
	a[0] = 2.0;
	a[1] = 1.0;
	a[2] = 5.0;
	a[3] = 7.0;
	
	dluDecompTB(a, dinv, l, u, 2, 0, 0);
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
	int testNum = 1;
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

	int lN;
	int lNT;
	int lNB;
	int lNK;
	for(int i = 0; i < testNum; i++){
		lN = params[i][0];
		lNT = params[i][1];
		lNB = params[i][2];
		lNK = params[i][3];
		/*
		dim3 threadPerBlock(lNT, lNT);
		dim3 blockPerGrid(lNK, lNK);
		
		float* h_a = initMat2DHelper(false, true, lN);
		float* h_b = initMat1DHelper(false, true, lN);
		float* h_x = initMat1DHelper(false, true, lN);
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

		for(int j = 0; j < lN * lN; j++){
			h_a[j] = avals[0][j];
		}
		
		for(int j = 0; j < lN; j++){
			h_b[j] = bvals[0][j];
			h_x[j] = xvals[0][j];
		}

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
		*/
		clock_t begin = clock();
		jacobiMethodTB(avals[i], bvals[i], xvals[i], lN, lNT, lNB, lNK);
		//cudaMemcpy(h_x, d_x, size2, cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		clock_t end = clock();
		
		for(int j = 0; j < lN; j++){
			if(xvals[i][j] != answers[i][j]){
				printf("Jacobi Method test failed\n");
				printMat1DHelper(xvals[i], lN);
				//exit(1);
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
