#include <iostream>
#include<stdio.h>
#include <vector>
#include "header files/matrix_computation.h"

int main() {

    // ===== DATA =====
    std::vector<std::vector<float>> X = { {1, 2, 3, 4} };
    std::vector<std::vector<float>> Y = { {2, 4, 6, 8} };

    int m = X[0].size();

    // ===== PARAMETERS =====
    int h = 3; // hidden neurons

    std::vector<std::vector<float>> W1 = { {0.5f}, {0.2f}, {-0.3f} };
    std::vector<float> b1 = { 0.0f, 0.0f, 0.0f };

    std::vector<std::vector<float>> W2 = { {0.4f, -0.2f, 0.1f} };
    std::vector<float> b2 = { 0.0f };

    float lr = 0.05f;
    int epochs = 2000;

    for (int epoch = 0; epoch < epochs; epoch++) {

        // ===== FORWARD =====
        auto Z1 = bias(mat_mul(W1, X), b1);
        auto A1 = RELU(Z1);

        auto Z2 = bias(mat_mul(W2, A1), b2);
        auto A2 = Z2; // identity

        // ===== LOSS =====
        float loss = Loss(A2, Y);

        // ===== BACKWARD =====
        auto dA2 = dLoss(A2, Y);
        auto dZ2 = dA2;

        auto dW2_mat = dW(dZ2, A1);
        auto db2_vec = db(dZ2);

        auto dA1 = dA_prev(W2, dZ2);
        auto dZ1 = dRELU(dA1, Z1);

        auto dW1_mat = dW(dZ1, X);
        auto db1_vec = db(dZ1);

        // ===== UPDATE =====

        // W2
        for (int i = 0; i < W2.size(); i++) {
            for (int j = 0; j < W2[0].size(); j++) {
                W2[i][j] -= lr * dW2_mat[i][j];
            }
        }

        // b2
        for (int i = 0; i < b2.size(); i++) {
            b2[i] -= lr * db2_vec[i];
        }

        // W1
        for (int i = 0; i < W1.size(); i++) {
            for (int j = 0; j < W1[0].size(); j++) {
                W1[i][j] -= lr * dW1_mat[i][j];
            }
        }

        // b1
        for (int i = 0; i < b1.size(); i++) {
            b1[i] -= lr * db1_vec[i];
        }

        // ===== PRINT =====
        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch
                << " Loss: " << loss << std::endl;
        }
    }

    return 0;
}
