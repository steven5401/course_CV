## How to build
Using CMake >= 2.8
```
mkdir build && cd build
cmake ..
make
```
Using g++
```
g++ main.cpp -o hw1 `pkg-config --cflags --libs opencv`
```
