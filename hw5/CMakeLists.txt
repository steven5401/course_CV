cmake_minimum_required(VERSION 3.2)
set (CMAKE_CXX_STANDARD 11)
project( hw5 )
find_package( OpenCV REQUIRED )
add_executable( hw5 main.cpp )
target_link_libraries( hw5 ${OpenCV_LIBS} )