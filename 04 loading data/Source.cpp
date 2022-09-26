#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

int main() {

	//--------- loading data -------------//
	/*
	* Before we were using set and setr to get data into vectors on the fly,
	* however there are better ways
	*/

	//-------- load ---------------//
	float* data = (float*)malloc(8 * sizeof(float));
	for (int lane = 0; lane < 8; ++lane) {
		data[lane] = lane;
	}
	__m256 mySIMDArray = _mm256_load_ps(data);
	free(data);
	float* SIMDdata = (float*)&mySIMDArray;
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- load unaligned ---------------//
	/*
	* I fudged the last example a little, the load function
	* should actually take data which is properly alligned to memory
	* (otherwise results may be unpredictable.)
	* 
	* loadu, on the other hand, will handle a regular malloc-ed block just fine.
	*/
	data = (float*)malloc(8 * sizeof(float));
	for (int lane = 0; lane < 8; ++lane) {
		data[lane] = lane;
	}
	mySIMDArray = _mm256_loadu_ps(data);
	free(data);
	SIMDdata = (float*)&mySIMDArray;
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance
	//setr
	data = (float*)malloc(8 * sizeof(float));
	for (int lane = 0; lane < 8; ++lane) {
		data[lane] = lane;
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000000; ++i) {
		mySIMDArray = _mm256_setr_ps(
			data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
		);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "load via setr took " << duration.count() << " ms." << std::endl;
	//load
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000000; ++i) {
		mySIMDArray = _mm256_load_ps(data);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "load function took " << duration.count() << " ms." << std::endl;
	//loadu
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000000; ++i) {
		mySIMDArray = _mm256_loadu_ps(data);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "loadu function took " << duration.count() << " ms." << std::endl;
	free(data);
	return 0;
}