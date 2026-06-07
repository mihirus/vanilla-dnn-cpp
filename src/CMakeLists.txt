cmake_minimum_required(VERSION 3.28)

find_package(Catch2 3 REQUIRED)

add_executable(mnist_loader_test mnist_loader_test.cpp)
target_link_libraries(mnist_loader_test
    PRIVATE
        Catch2::Catch2WithMain
        mnist_loader
)

add_executable(dnn_test dnn_test.cpp)
target_link_libraries(dnn_test
    PRIVATE
        Catch2::Catch2WithMain
        dnn
        mnist_loader
)

add_executable(mlp mlp.cpp)
target_link_libraries(mlp
    PRIVATE
        dnn
        mnist_loader
)