## How to build
Using CMake >= 2.8
```
mkdir build && cd build
cmake ..
make
```
Using g++
```
g++ --std=c++11 `pkg-config --cflags --libs opencv` main.cpp -o hw1
```
