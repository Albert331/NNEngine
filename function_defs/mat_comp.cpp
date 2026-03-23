#include "../header files/matrix_computation.h"

inline float square(float x) {
	return x * x;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix){
	int row = matrix.size();
	int col = matrix[0].size();

	std::vector<std::vector<float>> trans(col, std::vector<float>(row));

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			trans[j][i] = matrix[i][j];
		}
	}

	return trans;
}


std::vector<std::vector<float>> mat_mul(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2) {
	
	if (matrix1.empty() || matrix2.empty()) return {};
	
	int a = matrix1[0].size();
	int b = matrix2.size();
	float sum = 0;

	if (a == b) {
		a = matrix1.size();
		b = matrix2[0].size();
		std::vector<std::vector<float>> mat3(a, std::vector<float>(b));

		for (int i = 0; i < a; i++) {
			for (int j = 0; j < b;  j++) {
				for (int k =0 ; k < matrix1[0].size(); k++) {
					sum += matrix1[i][k] * matrix2[k][j];
				}
				mat3[i][j] = sum;
				sum = 0;
			}
		}
		return mat3;
	}
	else {
		std::cout << "nah fam not possible";
		return {};
	}
}


std::vector<std::vector<float>> bias(std::vector<std::vector<float>>& matrix1,const std::vector<float>& bias) {
	if (matrix1.empty() || bias.empty()) return {};
	
	int row1 = matrix1.size();
	int col1 = matrix1[0].size();
	int row2 = bias.size();

	if (row2 != row1) {
		std::cout << "sizes do not match";
		return matrix1;
	}

	for (int i = 0; i < row1; i++) {
		for (int j = 0; j < col1; j++) {
			matrix1[i][j] += bias[i];
		}
	}
	return matrix1;
}


std::vector<std::vector<float>> RELU(std::vector<std::vector<float>>& matrix) {
	
	if (matrix.empty()) return {};

	int row = matrix.size();
	int col = matrix[0].size();

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			matrix[i][j] = std::max(0.0f, matrix[i][j]);
		}
	}
	return matrix;

}

std::vector<std::vector<float>> dRELU(const std::vector<std::vector<float>>& dA,const std::vector<std::vector<float>>& z) {

	if (dA.empty() || z.empty()) return {};

	int row = z.size();
	int col = z[0].size();
	std::vector<std::vector<float>> dz(row, std::vector<float>(col));

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (z[i][j] > 0) {
				dz[i][j] = dA[i][j];
			}
			else {
				dz[i][j] = 0.0f;
			}
			
		}
	}
	return dz;

}


float Loss(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2) {
	float loss = 0.0f;
	int row = matrix1.size();
	int col = matrix2[0].size();
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			loss += square(matrix1[i][j] - matrix2[i][j]);
		}
	}
	return loss / (row*col);

}

std::vector<std::vector<float>> dLoss(const std::vector<std::vector<float>>& matrix1, const std::vector<std::vector<float>>& matrix2) {
	int row = matrix1.size();
	int col = matrix1[0].size();
	std::vector<std::vector<float>> dloss(row, std::vector<float>(col));
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			dloss[i][j] =(2.0f/(row*col))* (matrix1[i][j]- matrix2[i][j]);
		}
	}
	return dloss;
}

std::vector<std::vector<float>> dW(const std::vector<std::vector<float>>& dz,const std::vector<std::vector<float>>& a) {
	std::vector<std::vector<float>> dw= mat_mul(dz, transpose(a));
	int m = dz[0].size();

	for (auto& row : dw) {
		for (auto& val : row) {
			val /= m;
		}
	}
	return dw;
}

std::vector<float> db(const std::vector<std::vector<float>>& dz) {
	std::vector<float> db(dz.size(), 0.0f);

	for (int i = 0; i < dz.size(); i++) {
		float sum = 0.0f;
		for (int j = 0; j < dz[0].size(); j++) {
			sum += dz[i][j];
		}
		db[i] = sum / dz[0].size(); 
	}
	return db;
}

std::vector<std::vector<float>> dA_prev(const std::vector<std::vector<float>>& W,const std::vector<std::vector<float>>& dZ) {
	return mat_mul(transpose(W), dZ);
}