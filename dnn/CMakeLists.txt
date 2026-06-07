cmake_minimum_required(VERSION 3.28)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(dnn)

target_sources(dnn
    PRIVATE
        dnn.cpp

    PUBLIC
        FILE_SET HEADERS
        FILES
            dnn.h
)

target_link_libraries(dnn PUBLIC Eigen3::Eigen)
target_compile_features(dnn PUBLIC cxx_std_20)
