#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include<vector>
#include<iostream>
#include<algorithm>



std::vector<std::vector<float>> mat_mul(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2); 

std::vector<std::vector<float>> bias(std::vector<std::vector<float>>& matrix1 , const std::vector<float>& bias);

std::vector<std::vector<float>> RELU(std::vector<std::vector<float>>& matrix1);

float Loss(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2);

std::vector<std::vector<float>> dLoss(const std::vector<std::vector<float>>& matrix1, const std::vector<std::vector<float>>& matrix2);

std::vector<std::vector<float>> dRELU(const std::vector<std::vector<float>>& dA, const std::vector<std::vector<float>>& z);

std::vector<std::vector<float>> dW(const std::vector<std::vector<float>>& dz, const std::vector<std::vector<float>>& a);

std::vector<float> db(const std::vector<std::vector<float>>& dz);

std::vector<std::vector<float>> dA_prev(const std::vector<std::vector<float>>& W, const std::vector<std::vector<float>>& dZ);

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);

#endif