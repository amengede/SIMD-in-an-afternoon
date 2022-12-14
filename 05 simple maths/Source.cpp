#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

int main() {

	//--------- Simple maths -------------//
	float* data1 = (float*)malloc(8 * sizeof(float));
	data1[0] = 1;
	data1[1] = 2;
	data1[2] = 3;
	data1[3] = 4;
	data1[4] = 5;
	data1[5] = 6;
	data1[6] = 7;
	data1[7] = 8;
	float* data2 = (float*)malloc(8 * sizeof(float));
	data2[0] = 101;
	data2[1] = 102;
	data2[2] = 103;
	data2[3] = 104;
	data2[4] = 105;
	data2[5] = 106;
	data2[6] = 107;
	data2[7] = 108;
	__m256 vector1 = _mm256_loadu_ps(data1);
	__m256 vector2 = _mm256_loadu_ps(data2);

	float* data3 = (float*)malloc(8 * sizeof(float));

	//-------- addition ---------------//
	__m256 result = _mm256_add_ps(vector1, vector2);
	float* SIMDdata = (float*)&result;
	std::cout << "My Addition Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance
	//addition, regular
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			data3[lane] = data1[lane] + data2[lane];
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular addition took " << duration.count() << " ms." << std::endl;
	//addition, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_add_ps(vector1, vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD addition took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	//-------- subtraction ---------------//
	result = _mm256_sub_ps(vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Subtraction Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//Let's test performance
	//subtraction, regular
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			data3[lane] = data1[lane] - data2[lane];
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular subtraction took " << duration.count() << " ms." << std::endl;
	//subtraction, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_sub_ps(vector1, vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD subtraction took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	//-------- horizontal addition ---------------//
	result = _mm256_hadd_ps(vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Subtraction Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- add sub ---------------//
	result = _mm256_addsub_ps(vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Subtraction Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- mul ---------------//
	result = _mm256_mul_ps(vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Multiplication Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;
	//Let's test performance
	//subtraction, regular
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			data3[lane] = data1[lane] * data2[lane];
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular multiplication took " << duration.count() << " ms." << std::endl;
	//subtraction, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_mul_ps(vector1, vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD multiplication took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	//-------- mullo ---------------//
	__m256i intVec1 = _mm256_setr_epi16(3000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	__m256i intVec2 = _mm256_setr_epi16(3000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	// 3000*3000 will trigger a 16 bit overflow, with low short = 21568
	__m256i intResult = _mm256_mullo_epi16(intVec1, intVec2);
	short* shortData = (short*)&intResult;
	std::cout << "My low multiplication Result: ";
	for (int lane = 0; lane < 16; ++lane) {
		std::cout << shortData[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- mulhi ---------------//
	// 3000*3000 will trigger a 16 bit overflow, with high short = 137
	intResult = _mm256_mulhi_epi16(intVec1, intVec2);
	shortData = (short*)&intResult;
	std::cout << "My low multiplication Result: ";
	for (int lane = 0; lane < 16; ++lane) {
		std::cout << shortData[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- div ---------------//
	result = _mm256_div_ps(vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Division Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;
	//Let's test performance
	//division, regular
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			data3[lane] = data1[lane] / data2[lane];
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular division took " << duration.count() << " ms." << std::endl;
	//division, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_div_ps(vector1, vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD division took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	//-------- fused multiply and add ---------------//
	/*
	* fmadd(a,b,c) -> a*b + c
	*/
	__m256 vector3 = _mm256_setr_ps(-1, 2, -3, 4, -5, 6, -7, 8);
	result = _mm256_fmadd_ps(vector3, vector1, vector2);
	SIMDdata = (float*)&result;
	std::cout << "My Multiplication Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;
	//Let's test performance
	//individual operations
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000000; ++i) {
		result = _mm256_add_ps(_mm256_mul_ps(vector3,vector1), vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "individual operations took " << duration.count() << " ms." << std::endl;
	//subtraction, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000000; ++i) {
		result = _mm256_fmadd_ps(vector3, vector1, vector2);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Fused operations took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	free(data1);
	free(data2);
	free(data3);

	return 0;
}