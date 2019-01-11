cmake_minimum_required(VERSION 3.10)

project(proj3ctf1lter)

find_package(OpenCV REQUIRED)

# Remove -DCPU_ONLY in order to use GPU
SET(CMAKE_CXX_FLAGS "-DUSE_OPENCV")
SET(CMAKE_EXE_LINKER_FLAGS "-lglog -lboost_system -L/usr/lib/caffe/ -lcaffe")

file(GLOB proj3ctf1lter_SRC 
	"./src/main.cpp"
	"./src/net.hpp"
	"./src/net.cpp"
)

add_executable(main ${proj3ctf1lter_SRC})
target_link_libraries(main ${CMAKE_EXE_LINKER_FLAGS} ${OpenCV_LIBS})
