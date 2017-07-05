/**
 * @file   : DNN_playground_unified.cu
 * @brief  : Deep Neural Network (DNN) playground, with unified memory
 * uses CUDA Unified Memory (Management)
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170506
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * nvcc -arch='sm_61' -lcudnn DNN_playground_unified.cu -o DNN_playground_unified.exe
  * -lcudnn flag says 'l' for library to include, and so you'd want to include 'cudnn'
  * */
#include <iostream>

#include <cudnn.h>

__device__ __managed__ float A[3*4]; 
__device__ __managed__ float B[3*4];
__device__ __managed__ float C[3*4];

int main() {
	// cudnnDataType_t, enumerated type indicating data type which tensor descriptor refers to
	std::cout << "CUDNN_DATA_FLOAT  : " << CUDNN_DATA_FLOAT << std::endl;
	std::cout << "CUDNN_DATA_DOUBLE : " << CUDNN_DATA_DOUBLE << std::endl;
	std::cout << "CUDNN_DATA_HALF   : " << CUDNN_DATA_HALF << std::endl;
	std::cout << "CUDNN_DATA_INT8   : " << CUDNN_DATA_INT8 << std::endl;
	std::cout << "CUDNN_DATA_INT32  : " << CUDNN_DATA_INT32 << std::endl;
	std::cout << "CUDNN_DATA_INT8x4 : " << CUDNN_DATA_INT8x4 << std::endl;

	// cudnnOpTensorOp_t, enumerated type indicating tensor operation used by cudnnOpTensor() 
	std::cout << "CUDNN_OP_TENSOR_ADD : " << CUDNN_OP_TENSOR_ADD << std::endl;
	std::cout << "CUDNN_OP_TENSOR_MUL : " << CUDNN_OP_TENSOR_MUL << std::endl;
	std::cout << "CUDNN_OP_TENSOR_MIN : " << CUDNN_OP_TENSOR_MIN << std::endl;
	std::cout << "CUDNN_OP_TENSOR_MAX : " << CUDNN_OP_TENSOR_MAX << std::endl;

	// cudnnReduceTensorOp_t, enumerated type indicating tensor operation used by cudnnReduceTensor() 
	std::cout << "CUDNN_REDUCE_TENSOR_ADD   : " << CUDNN_REDUCE_TENSOR_ADD << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_MUL   : " << CUDNN_REDUCE_TENSOR_MUL << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_MIN   : " << CUDNN_REDUCE_TENSOR_MIN << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_MAX   : " << CUDNN_REDUCE_TENSOR_MAX << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_AMAX  : " << CUDNN_REDUCE_TENSOR_AMAX << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_AVG   : " << CUDNN_REDUCE_TENSOR_AVG << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_NORM1 : " << CUDNN_REDUCE_TENSOR_NORM1 << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_NORM2 : " << CUDNN_REDUCE_TENSOR_NORM2 << std::endl;

	// cudnnReduceTensorIndices_t, enumerated type used to indicate whether indices are to be computed by 
	// cudnnReduceTensor() routine; used as field for cudnnReduceTensorDescriptor_t descriptor
	std::cout << "CUDNN_REDUCE_TENSOR_NO_INDICES        : " << CUDNN_REDUCE_TENSOR_NO_INDICES << std::endl;
	std::cout << "CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : " << CUDNN_REDUCE_TENSOR_FLATTENED_INDICES << std::endl;
	
	// cudnnIndicesType_t, enumerated type used to indicate data type for indices to be computed by 
	// cudnnReduceTensor(), used as field for cudnnReduceTensorDescriptor_t descriptor
	std::cout << "CUDNN_32BIT_INDICES : " << CUDNN_32BIT_INDICES << std::endl;
	std::cout << "CUDNN_64BIT_INDICES : " << CUDNN_64BIT_INDICES << std::endl;
	std::cout << "CUDNN_16BIT_INDICES : " << CUDNN_16BIT_INDICES << std::endl;
	std::cout << "CUDNN_8BIT_INDICES  : " << CUDNN_8BIT_INDICES << std::endl;


	// cudnnNanPropagation_t, enumerated type indicating if routines should propagate Nan numbers
	std::cout << "CUDNN_NOT_PROPAGATE_NAN : " << CUDNN_NOT_PROPAGATE_NAN << std::endl;
	std::cout << "CUDNN_PROPAGATE_NAN     : " << CUDNN_PROPAGATE_NAN << std::endl;
	


	size_t version_cuDNN;
	version_cuDNN = cudnnGetVersion();
	std::cout << "cuDNN version " << version_cuDNN << std::endl; 
	
	size_t version_toolkit; 
	version_toolkit = cudnnGetCudartVersion();
	std::cout << "CUDA Toolkit version " << version_toolkit << std::endl;
	
	// Create cuDNN handles
	cudnnHandle_t cudnnHandle; 
	cudnnCreate(&cudnnHandle); 
	
	// generic Tensor descriptor
	cudnnTensorDescriptor_t XDesc;
	cudnnCreateTensorDescriptor(&XDesc);
	
	// initializes previously created generic Tensor descriptor object
	// "When working with lower dimensional data, 
	// it is recommended that the user create a 4D tensor, and 
	// set the size along unused dimensions to 1." cuDNN API reference
	constexpr const int nbDims_X = 4; // dimension of tensor X
	constexpr const int dx = 3;
	constexpr const int dy = 4;
	constexpr const int dimA_X[nbDims_X] {2,dx,dy,1}; // array of dim. nbDims that contain size of tensor for every dim.
	
	/* 
	 * dealing with stride
	 * now strideA, strideA_X, in this case is the array of dimension nbDims that 
	 * contain the stride of the tensor for every dimension.
	 * We have to deal with memory layout and seeking a contiguous memory layout. 
	 * 
	 * let's try this.  For number of input samples m, with dx x dy features, and so 
	 * each sample (or example) has a total of dx*dy features, for dimension sizes
	 * (m,dx,dy)
	 * (notice the order)
	 * the striding would then be 
	 * dx*dy, 1, dx
	 * (I wanted "row-order" ordering, so that each element in a row is laid out 
	 * contiguously in memory.  
	 * */
	
	constexpr const int strideA_X[nbDims_X] { dx*dy, 1, dx, 2*dx*dy } ;
	
	cudnnSetTensorNdDescriptor( XDesc, CUDNN_DATA_FLOAT, nbDims_X, dimA_X, strideA_X) ; 
	
	/* ************************************************************** */
	/* ****************   (Basic) Linear Algebra   ****************** */
	/* ************************************************************** */
	
	// Let's try some basic Linear Algebra
	// generic Tensor descriptor
	cudnnTensorDescriptor_t ADesc;
	cudnnCreateTensorDescriptor(&ADesc);
	cudnnTensorDescriptor_t BDesc;
	cudnnCreateTensorDescriptor(&BDesc);

		
	constexpr const int N1 = 3; 
	constexpr const int N2 = 4; 
	constexpr const int dimsizesA[nbDims_X] { N1,N2,1,1}; // array of dim. nbDims that contain size of matrix
	constexpr const int stridesA[nbDims_X] { N2, 1, N1*N2, N1*N2 };
	
	cudnnSetTensorNdDescriptor( ADesc, CUDNN_DATA_FLOAT, nbDims_X, dimsizesA, stridesA ) ;
	cudnnSetTensorNdDescriptor( BDesc, CUDNN_DATA_FLOAT, nbDims_X, dimsizesA, stridesA ) ;
	
	// initialize boilerplate initial values for matrices A,B
	for (int idx = 0; idx < N1*N2; idx++) { 
		A[idx] = ((float) idx);
		B[idx] = idx + 10.f;
		C[idx] = 0.f;
	}
	float alpha = 1.f;
	float beta  = 1.f;
	
	/* sanity check: printing out values for our matrices A,B */
	std::cout << "Initially : " << std::endl;
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	
	cudnnAddTensor( cudnnHandle, &alpha, ADesc, &A, &beta, BDesc, &B);
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 
	

	std::cout << "Attempt at cudnnAddTensor #1 " << std::endl;
	/* sanity check: printing out values for our matrices A,B */
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	
	std::cout << " Try using the same description : " << std::endl;
	cudnnAddTensor( cudnnHandle, &alpha, ADesc, &A, &alpha, ADesc, &B);
	
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 

	/* sanity check: printing out values for our matrices A,B,C */
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;



	cudnnOpTensorDescriptor_t MatrixRingAdditionDesc; 
	cudnnCreateOpTensorDescriptor(&MatrixRingAdditionDesc);	
	cudnnSetOpTensorDescriptor(MatrixRingAdditionDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, 
		CUDNN_NOT_PROPAGATE_NAN); 
	
	std::cout << "Initially, this is the value of C : " << std::endl;
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << C[j + i*N2] << " "; }
		std::cout << std::endl; }

	
	cudnnOpTensor(cudnnHandle,MatrixRingAdditionDesc,&alpha,ADesc,&A,&alpha,ADesc,&B,&alpha,ADesc,&C);
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 

	std::cout << " After cudnnOpTensor, this is A,B,C  : " << std::endl;
	
	/* sanity check: printing out values for our matrices A,B */
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;



	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << C[j + i*N2] << " "; }
		std::cout << std::endl; }

	cudnnOpTensorDescriptor_t MatrixRingMultiplyDesc; 
	cudnnCreateOpTensorDescriptor(&MatrixRingMultiplyDesc);	
	cudnnSetOpTensorDescriptor(MatrixRingMultiplyDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, 
		CUDNN_NOT_PROPAGATE_NAN); 

	cudnnOpTensor(cudnnHandle,MatrixRingMultiplyDesc,&alpha,ADesc,&A,&alpha,ADesc,&B,&alpha,ADesc,&C);
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 

	std::cout << " After cudnnOpTensor by CUDNN_OP_TENSOR_MUL (Multiply), this is A,B,C  : " << std::endl;
	
	/* sanity check: printing out values for our matrices A,B */
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;



	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << C[j + i*N2] << " "; }
		std::cout << std::endl; }
	
	std::cout << " It's interesting that CUDNN_OP_TENSOR_MUL is actually a triple operation, not a binary operation. " << std::endl << 
		" Set the value of C to 0 " << std::endl; 
	/* This is interesting.  For 
	 * CUDNN_OP_TENSOR_MUL
	 * it's actually a triple operation A,B,C |-> A * B + C, with element-wise multiplication here.  
	 * */

	float identity_reals = 0.f ; 
	cudnnSetTensor(cudnnHandle, ADesc, &C, &identity_reals);  
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 

	 
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << C[j + i*N2] << " "; }
		std::cout << std::endl; }
	 
	cudnnOpTensor(cudnnHandle,MatrixRingMultiplyDesc,&alpha,ADesc,&A,&alpha,ADesc,&B,&alpha,ADesc,&C);
	// probably need a cudaDeviceSynchronize, otherwise it won't know to print first or do operation
	// take out cudaDeviceSynchronize in "production" code, because we only need it here for printing stuff out
	cudaDeviceSynchronize(); 

	std::cout << " After cudnnOpTensor by CUDNN_OP_TENSOR_MUL (Multiply), this is A,B,C  : " << std::endl;
	
	/* sanity check: printing out values for our matrices A,B */
	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << A[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << B[j + i*N2] << " "; }
		std::cout << std::endl; }

	std::cout << std::endl;

	for (int i=0; i<N1; i++) {
		for (int j=0;j<N2;j++){ 
			std::cout << C[j + i*N2] << " "; }
		std::cout << std::endl; }

	/* 
	 * Reduce 
	 * */

	cudnnReduceTensorDescriptor_t reduceAddDesc; 
	cudnnCreateReduceTensorDescriptor(&reduceAddDesc);	
	cudnnSetReduceTensorDescriptor(reduceAddDesc, CUDNN_REDUCE_TENSOR_ADD, 
		CUDNN_DATA_FLOAT, 
		CUDNN_NOT_PROPAGATE_NAN, 
		CUDNN_REDUCE_TENSOR_NO_INDICES, 
		CUDNN_32BIT_INDICES ); 



	
	
	cudnnActivationDescriptor_t psi_desc; 

	cudnnCreateActivationDescriptor(&psi_desc) ;
	
	// destroy Tensor Descriptors, and then cuDNN handle
	cudnnDestroyTensorDescriptor( XDesc);
	cudnnDestroyTensorDescriptor( ADesc);  
	cudnnDestroyTensorDescriptor( BDesc);  

	cudnnDestroy(cudnnHandle);  

	cudnnDestroyOpTensorDescriptor(MatrixRingAdditionDesc);
	cudnnDestroyOpTensorDescriptor(MatrixRingMultiplyDesc);

	cudnnDestroyReduceTensorDescriptor(reduceAddDesc);


}
  
  
