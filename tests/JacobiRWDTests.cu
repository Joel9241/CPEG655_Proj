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
	dim3 threadPerBlock(3, 1);
	dim3 blockPerGrid(1, 1);
	
	float *h_a = initMat2DHelper(false, true, 3);
	float *h_b = initMat1DHelper(false, true, 3);
	float *h_c = initMat1DHelper(false, true, 3);
	
	float *d_a = initMat2DHelper(false, false, 3);
	float *d_b = initMat1DHelper(false, false, 3);
	float *d_c = initMat1DHelper(false, false, 3);	

	for(int i = 0; i < 9; i++){
		h_a[i] = i + 1;
	}
	for(int i = 0; i < 3; i++){
		h_b[i] = i + 1;
	}

	size_t size1 = N * N * sizeof(float);
	size_t size2 = N * sizeof(float);

	cudaMemcpy(d_a, h_a, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size2, cudaMemcpyHostToDevice);

	multiplyMats2D1DTB<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c, 3, 1, 1);
	cudaMemcpy(h_c, d_c, size2, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if((h_c[0] != 14) || (h_c[1] != 32) || (h_c[2] != 50)){
		printf("multiplyMats2D1D unsuccessfull\n");
		printMat1DHelper(h_c, 3);
		exit(1);
	}
}

void multiplyMats2DTest(){
	dim3 threadPerBlock(3, 3);
	dim3 blockPerGrid(1, 1);
	
	float *h_a = initMat2DHelper(false, true, 3);
	float *h_b = initMat2DHelper(false, true, 3);
	float *h_c = initMat2DHelper(false, true, 3);
	
	float *d_a = initMat2DHelper(false, false, 3);
	float *d_b = initMat2DHelper(false, false, 3);
	float *d_c = initMat2DHelper(false, false, 3);	

	for(int i = 0; i < 9; i++){
		h_a[i] = i + 1;
	}
	for(int i = 0; i < 9; i++){
		h_b[i] = i + 1;
	}

	size_t size = N * N * sizeof(float);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	multiplyMats2DTB<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c, 3, 1, 1);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	float ctb[9] = {30, 36, 42, 66, 81, 96, 102, 126, 150};
	for(int i = 0; i < 9; i++){
		if(h_c[i] != ctb[i]){
			printf("multiplyMats2D1D unsuccessfull\n");
			printMat2DHelper(h_c, 3);
			exit(1);
		}
	}
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
	//dluTest();
	//multiplyMats2D1DTest();
	multiplyMats2DTest();
	//cuda_hello<<<1, 1>>>();
	

	/*
	jacobiMethodTest();
	*/
	printf("If you got here all tests passed\n");
	return 0;
}
