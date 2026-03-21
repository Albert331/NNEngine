#include <iostream>
#include<stdio.h>
#include <vector>
#include "header files/matrix_comp.h"

int main() {
	std::vector<std::vector<float>> X = { {1.0f, 2.0f},{3.0f, 4.0f} };

	std::vector<std::vector<float>> W1 = { {0.5f, -0.2f, 0.1f}, {0.3f, 0.8f, -0.5f} };

	std::vector<float> b1 = { 0.1f, 0.2f, 0.3f };

	std::vector<std::vector<float>> XW1 = mat_mul(X, W1);

	std::vector<std::vector<float>> WB = bias(XW1, b1);

	std::vector<std::vector<float>> WA = RELU(WB);


	for (int i = 0; i < XW1.size(); i++) {
		for (int j = 0; j < XW1[0].size(); j++) {
			std::cout << WA[i][j] << " ";
		}
		std::cout << std::endl;
	}
	return 0;
}
