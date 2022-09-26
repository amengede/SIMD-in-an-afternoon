#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

int main() {

	// __m256 : a chunk of 256 bits, equivalent to an array of 8 floats

	//--------- setzero -------------//
	//Vanilla array initialization.
	float myArray[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	std::cout << "My Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << myArray[lane] << ", ";
	}
	std::cout << std::endl;

	//SIMD method: _mm256_setzero_ps() / _mm_setzero_ps()
	/*
	* (SIMD stands for "Single instruction, multiple data. But no, I don't know
	* what all these ms stand for in the function names).
	* set packed single precision floats to zero.
	*/
	__m256 mySIMDArray = _mm256_setzero_ps();
	float* data = (float*)&mySIMDArray; //get the address of the array, then treat it
										//like the address of a float.
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << data[lane] << ", ";
	}
	std::cout << std::endl;
	
	//Let's test performance.
	// 
	//standard
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			myArray[lane] = 0.0f;
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of 0.0f took " << duration.count() << " ms." << std::endl;
	//
	//SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		mySIMDArray = _mm256_setzero_ps();
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of 0.0f took " << duration.count() << " ms." << std::endl;
	std::cout << "------------" << std::endl;

	//--------- set1 -------------//
	//Vanilla array initialization.
	double myArray2[4] = { 10.0, 10.0, 10.0, 10.0};
	std::cout << "My Array: ";
	for (int lane = 0; lane < 4; ++lane) {
		std::cout << myArray2[lane] << ", ";
	}
	std::cout << std::endl;

	//SIMD method: _mm256_set1_pd(double value) / _mm_set1_pd(double value)
	/*
	* set packed double precision floats to a uniform value.
	*/
	__m256d mySIMDArray2 = _mm256_set1_pd(10.0);
	double* data2 = (double*)&mySIMDArray2; //get the address of the array, then treat it
	//like the address of a double
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane) {
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance.
	// 
	//standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 4; ++lane) {
			myArray2[lane] = 10.0;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of 10.0 took " << duration.count() << " ms." << std::endl;
	//
	//SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		mySIMDArray2 = _mm256_set1_pd(10.0);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of 10.0 took " << duration.count() << " ms." << std::endl;
	std::cout << "------------" << std::endl;

	//--------- set -------------//
	//Vanilla array initialization.
	double myArray3[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	std::cout << "My Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << myArray3[lane] << ", ";
	}
	std::cout << std::endl;

	//SIMD method: _mm256_set_epi32(int val1, ...)
	/*
	* set a vector's contents to store the eight provided ints
	*/
	__m256i mySIMDArray3 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8); 
			// the main data types are __m256, __m256d, __m256i,
			// regardless of what gets stored
	int* data3 = (int*)&mySIMDArray3; //get the address of the array, then treat it
	//like the address of an int
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << data3[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance.
	// 
	//standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			myArray3[lane] = lane;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of integers took " << duration.count() << " ms." << std::endl;
	//
	//SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		mySIMDArray3 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of integers took " << duration.count() << " ms." << std::endl;
	std::cout << "------------" << std::endl;

	//--------- setr -------------//
	//Vanilla array initialization.
	double myArray4[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	std::cout << "My Array: ";
	for (int lane = 0; lane < 16; ++lane) {
		std::cout << myArray4[lane] << ", ";
	}
	std::cout << std::endl;

	//SIMD method: _mm256_setr_epi16(int val1, ...)
	/*
	* set a vector's contents to store the 16 provided ints,
	* loading numbers in reverse order
	*/
	__m256i mySIMDArray4 = _mm256_setr_epi16(
		1,  2,  3, 4,  5,  6,  7,  8,
		9, 10, 11,12, 13, 14, 15, 16);
	// the main data types are __m256, __m256d, __m256i,
	// regardless of what gets stored
	short* data4 = (short*)&mySIMDArray4; //get the address of the array, then treat it
	//like the address of a short
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 16; ++lane) {
		std::cout << data4[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance.
	// 
	//standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 16; ++lane) {
			myArray4[lane] = lane;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of integers took " << duration.count() << " ms." << std::endl;
	//
	//SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		mySIMDArray4 = _mm256_setr_epi16(
			1, 2, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of integers took " << duration.count() << " ms." << std::endl;
	std::cout << "------------" << std::endl;
	
	return 0;
}