#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>
#include <glm/glm.hpp>
#include <array>

int main() {

	//--------- Dot Products! -------------//
	std::array<glm::vec3, 8> vectors1 = { {
			glm::vec3(1.0f, 0.2f, 0.3f),
			glm::vec3(0.1f, -0.4f, -0.6f),
			glm::vec3(1.0f, 1.0f, 0.9f),
			glm::vec3(-0.2f, 1.0f, -1.2f),
			glm::vec3(1.0f, -0.8f, 1.0f),
			glm::vec3(-0.3f, 1.6f, 1.0f),
			glm::vec3(1.0f, 1.0f, 1.0f),
			glm::vec3(-0.4f, -3.2f, 1.0f),
	} };
	std::array<glm::vec3, 8> vectors2 = { {
			glm::vec3(-1.0f, -0.6f, 0.7f),
			glm::vec3(-0.4f, 1.2f, -1.4f),
			glm::vec3(-1.0f, 1.0f, 2.1f),
			glm::vec3(0.8f, 1.0f, -2.8f),
			glm::vec3(-1.0f, -1.6f, 1.0f),
			glm::vec3(1.2f, 1.6f, 1.0f),
			glm::vec3(-1.0f, 1.0f, 1.0f),
			glm::vec3(1.6f, -2.4f, 1.0f),
	} };
	float result;

	//-------- naive approach ---------------//
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			result = glm::dot(vectors1[lane], vectors2[lane]);
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "glm dot product took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	//-------- simd approach ---------------//
	//declare simd memory containers
	union { __m256 x1; float x_1[8]; };
	union { __m256 x2; float x_2[8]; };
	union { __m256 y1; float y_1[8]; };
	union { __m256 y2; float y_2[8]; };
	union { __m256 z1; float z_1[8]; };
	union { __m256 z2; float z_2[8]; };
	__m256 SIMDresult;

	//unpack data from vectors, pack into simd lanes
	for (int lane = 0; lane < 8; ++lane) {
		x_1[lane] = vectors1[lane].x;
		x_2[lane] = vectors2[lane].x;
		y_1[lane] = vectors1[lane].y;
		y_2[lane] = vectors2[lane].y;
		z_1[lane] = vectors1[lane].z;
		z_2[lane] = vectors2[lane].z;
	}

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		//compute dot product
		// x1*x2 + y1*y2 + z1*z2
		// = (x1 * x2 + [y1 * y2 + <z1 * z2>]), 2fmas, 1 mul
		SIMDresult = _mm256_fmadd_ps(x1, x2, _mm256_fmadd_ps(y1, y2, _mm256_mul_ps(z1, z2)));
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD dot product took " << duration.count() << " ms." << std::endl;
	std::cout << "---------------------" << std::endl;

	return 0;
}