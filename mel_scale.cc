// g++ -std=c++11 -I ./eigen-3.4.0/ mel_scale.cc -o mel_scale
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
 
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Matrix_int8;
 
int main() {
	// input data is 384000 audio samples of float
	// take a FFT: window size 2048, hop 512 samples for each window
	// yes the FFT accelerator aka Xilinx HLS FFT takes float
	Eigen::MatrixXf FFTOuput(1025, 750);
	auto FFTOuput_transposed = FFTOuput.transpose();
	FFTOuput = FFTOuput*FFTOuput_transposed; // square for power 
	FFTOuput = FFTOuput/700;
	//ref: https://stackoverflow.com/questions/33786662/apply-function-to-all-eigen-matrix-element
	FFTOuput = FFTOuput.unaryExpr([](float x){return x + 1;});
	FFTOuput = FFTOuput.unaryExpr([](float x){return std::log10(x);});
	FFTOuput = FFTOuput*2595;
	// skip some details of how mel-scale with 128 bins maps to FFTouput 	
	
	// after MFCC
	Eigen::MatrixXf MFCCAudio(128, 750);
	MFCCAudio.conservativeResize(1, 128*750); // reshape to a vector
	// std::cout << "MFCCAudio is of size " << MFCCAudio.rows() << "x" << MFCCAudio.cols() << std::endl;
	Matrix_int8 MFCCAudio_int8t = MFCCAudio.cast <int8_t> (); 
	// std::cout << "audio is of size " << MFCCAudio_int8t.rows() << "x" << MFCCAudio_int8t.cols() << std::endl;
}