## How to build
Using CMake >= 2.8
```
mkdir build && cd build
cmake ..
make
```
Or using g++
```
g++ --std=c++11 main.cpp `pkg-config --cflags --libs opencv` -o hw6
```
## How to use
Using redirection, then open that file
```
./hw6 [path-to-lena.bmp] > [path-to-save-stdout]
```