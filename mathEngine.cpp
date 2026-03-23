#include <iostream>
#include<stdio.h>
#include <vector>
#include "header files/matrix_computation.h"

int main() {

    

        // ===== DATA =====
        std::vector<std::vector<float>> X = { {1, 2, 3, 4} };
        std::vector<std::vector<float>> Y = { {2, 4, 6, 8} };

        // ===== PARAMETERS =====
        std::vector<std::vector<float>> W = { {0.5f} }; // 1x1
        std::vector<float> b = { 0.0f };

        float lr = 0.01f;
        int epochs = 1000;

        for (int epoch = 0; epoch < epochs; epoch++) {

            // ===== FORWARD =====
            auto Z = mat_mul(W, X);
            Z = bias(Z, b);

            auto A = Z; // identity activation

            // ===== LOSS =====
            float loss = Loss(A, Y);

            // ===== BACKWARD =====
            auto dA = dLoss(A, Y);
            auto dZ = dA; // identity derivative

            auto gradW = dW(dZ, X);
            auto gradb = db(dZ);

            // ===== UPDATE =====
            for (int i = 0; i < W.size(); i++) {
                for (int j = 0; j < W[0].size(); j++) {
                    W[i][j] -= lr * gradW[i][j];
                }
            }

            for (int i = 0; i < b.size(); i++) {
                b[i] -= lr * gradb[i];
            }

            // ===== PRINT =====
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch
                    << " Loss: " << loss
                    << " W: " << W[0][0]
                    << " b: " << b[0]
                    << std::endl;
            }

        }
        return 0;
    }

