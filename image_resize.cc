// g++ -std=c++11 -I ./eigen-3.4.0 image_resize.cc -o image_resize
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {	
	Eigen::Tensor<float, 3> input(1080, 720, 1);
	Eigen::Tensor<float, 2> kernel(2, 2);
	Eigen::Tensor<float, 3> output(540, 360, 1);

	//input.setRandom();
	input.setConstant(4);
	//std::cout << input << "\n";
	std::cout << "input dimensions\n";
	std::cout << "rows:" << input.dimension(0) << ",cols:" << input.dimension(1) << "\n";

	Eigen::array<ptrdiff_t, 2> dims({0, 1});  // Specify dim 0 and 1 

	// tensor slicing to take a 2*2 patch for max pooling
	Eigen::array<Eigen::Index, 3> extents = {2, 2, 1};
	for(int ch = 0; ch < 1; ch++){
		for(int i = 0; i < 4; i=i+2) {
			for(int j = 0; j < 4; j=j+2) {
				Eigen::array<Eigen::Index, 3> offsets = {i, j, ch};				
				Eigen::Tensor<float, 3> slice = input.slice(offsets, extents);
				Eigen::Tensor<float, 1> val = slice.maximum(Eigen::array<int, 2>({0, 1}));
				output(i/2, j/2, ch) = val(0);

				// auto a = input(i, j, ch)*kernel(0,0);
				// auto b = input(i+1, j, ch)*kernel(1,0);
				// auto c = input(i, j+1, ch)*kernel(0,1);
				// auto d = input(i+1, j+1, ch)*kernel(1,1);
				// auto val = (a + b + c + d)/4;
				// output(i/2, j/2, ch) = val; 
			}
		}
	}
	//std::cout << output << "\n";
	std::cout << "output dimensions\n";
	std::cout << "rows:" << output.dimension(0) << ",cols:" << output.dimension(1) << "\n";
}
