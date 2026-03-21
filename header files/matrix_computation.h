#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include<vector>
#include<iostream>
#include<algorithm>
#define square(x) ((x) * (x))


std::vector<std::vector<float>> mat_mul(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2); 

std::vector<std::vector<float>> bias(std::vector<std::vector<float>>& matrix1 , const std::vector<float>& bias);

std::vector<std::vector<float>> RELU(std::vector<std::vector<float>>& matrix1);

float Loss(const std::vector<std::vector<float>>& matrix1,const std::vector<std::vector<float>>& matrix2);







#endif