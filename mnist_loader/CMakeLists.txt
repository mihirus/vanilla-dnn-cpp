cmake_minimum_required(VERSION 3.28)

add_library(mnist_loader)

target_sources(mnist_loader
    PRIVATE
        mnist_loader.cpp
    
    PUBLIC
        FILE_SET HEADERS
        FILES
          mnist_loader.h
)

target_compile_features(mnist_loader PUBLIC cxx_std_20)