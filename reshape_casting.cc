// g++ -std=c++11 -I ./eigen-3.4.0/ reshape_casting.cc -o reshape_casting
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
 
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Matrix_int8;
 
int main() {
	Eigen::MatrixXf FFTOuput(16, 250);
	auto FFTOuput_transposed = FFTOuput.transpose();
	FFTOuput = FFTOuput*FFTOuput_transposed; // square for power 
	FFTOuput.conservativeResize(1, 16*250); // reshape to a vector
	Matrix_int8 FFTOuput_int8t = FFTOuput.cast <int8_t> (); 
	std::cout << "audio is of size " << FFTOuput_int8t.rows() << "x" << FFTOuput_int8t.cols() << std::endl;
}