#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

int main() {

	//--------- binding data -------------//
	/*
	* To interface with data, eg. streaming out to non-vectorized code,
	* we can do some pointer conversion
	*/
	__m256 mySIMDArray = _mm256_setzero_ps();
	float* data = (float*)&mySIMDArray; //get the address of the array, then treat it
										//like the address of a float.
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << data[lane] << ", ";
	}
	std::cout << std::endl;
	
	/*
	* This is good, there's another way: unions
	*/
	union {
		__m256d mySIMDArray2;
		double data2[4];
	}; //both mySIMDArray2 and data2 occupy the same memory.
	mySIMDArray2 = _mm256_setr_pd(3.5, -7.0, 2.0, 5.6);
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane) {
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	std::cout << "Modifying data..." << std::endl;
	for (int lane = 0; lane < 4; ++lane) {
		data2[lane] = lane;
	}

	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane) {
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	/*
	*	Unions and pointer conversion are both valid ways to read/write simd data,
	*	though obviously they suffer the same performance hit as regular array operations
	*/
	
	return 0;
}