// g++ -std=c++11 -I ./eigen-3.4.0/ mel_scale.cc -o mel_scale
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
 
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Matrix_int8;
 
int main() {
	uint32_t iterations = 2;//1024*1024;
	uint32_t batch_size = 2;	

	for(int iter = 0; iter < iterations; iter++){
		Eigen::Tensor<float, 3> FFTOuput(1025, 750, batch_size);
		Eigen::Tensor<float, 3> matmul(1025, 1025, batch_size);
		Eigen::Tensor<float, 3> constmat(1025, 1025, batch_size);	
		FFTOuput.setRandom();

		Eigen::array<Eigen::Index, 3> extents = {1, 0, 2};
		//auto FFTOuput_transposed = FFTOuput.transpose();
		Eigen::Tensor<float, 3> FFTOuput_transposed = FFTOuput.shuffle(extents); 
		//FFTOuput = FFTOuput*FFTOuput_transposed; // square for power 
		for(int batch = 0; batch < batch_size; batch++){
			Eigen::Tensor<float, 2> mat_a = FFTOuput.chip(batch,2);
			Eigen::Tensor<float, 2> mat_b = FFTOuput_transposed.chip(batch,2);
			Eigen::Tensor<float, 2> mat_ab = matmul.chip(batch,2);
			// std::cout << "mat_a:\n" << mat_a << std::endl;
			// std::cout << "mat_b:\n" << mat_b << std::endl;
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
			mat_ab = mat_a.contract(mat_b, product_dims);
			std::cout << mat_ab.NumDimensions << "\n";
		}
		constmat.setConstant(700);
		matmul = matmul/constmat;
		matmul = matmul.unaryExpr([](float x){return x + 1;});
		matmul = matmul.unaryExpr([](float x){return std::log10(x);});
		constmat.setConstant(2595);
		matmul = matmul*constmat;
		Eigen::Tensor<int8_t, 3>  MFCCAudio_int8t = matmul.cast <int8_t> (); 	
		Eigen::array<uint32_t, 3> three_dims{1025*1025, 1, batch_size};
		MFCCAudio_int8t = MFCCAudio_int8t.reshape(three_dims);
		std::cout << MFCCAudio_int8t.NumDimensions << "\n";
	}
}