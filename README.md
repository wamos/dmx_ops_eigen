### DMX operations we used in benchmarks

We need eigen 3.4.x to compile these C++ code.  
You can download Eigen 3.4.0 here [http://eigen.tuxfamily.org/index.php?title=Main_Page#Download]  

Each .cc file has a line like the following to help you compile the code.
``
g++ -std=c++11 -I ./eigen-3.4.0/ {source_file}.cc -o {exec_file}

``
The code requires C++11 as we used Lambda in some of the operations.
