#include <iostream>
#include<stdio.h>
#include <vector>
#include "header files/matrix_computation.h"

int main() {

	std::vector<std::vector<float>> X = { {1.0f, 2.0f},{3.0f, 4.0f} };
	std::vector<std::vector<float>> W1 = { {0.5f, -0.2f, 0.1f}, {0.3f, 0.8f, -0.5f} };
	std::vector<float> b1 = { 0.1f, 0.2f, 0.3f };

	std::vector<std::vector<float>> XW1 = mat_mul(X, W1);
	std::vector<std::vector<float>> WB = bias(XW1, b1);
	std::vector<std::vector<float>> WA = RELU(WB);

	std::vector<std::vector<float>> W2 = { {0.7f },{-0.3f},{0.5f} };
	std::vector<float> b2 = { 0.2f };

	std::vector<std::vector<float>> XW2 = mat_mul(WA, W2);
	std::vector<std::vector<float>> WB2 = bias(XW2, b2);
	std::vector<std::vector<float>> WA2 = RELU(WB2);

	

	std::vector<std::vector<float>> expected = { {0.1f},{0.0f} };
	float losss = Loss(WA2, expected);
	std::cout << WA2[0][0] <<" " << WA2[1][0] << std::endl;
	std::cout << losss;






	return 0;
}
