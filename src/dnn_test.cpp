#include "dnn.h"
#include "mnist_loader.h"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

std::string train_imgs;
std::string train_lbls;
std::string test_imgs;
std::string test_lbls;

TEST_CASE("DNN") {
  std::vector<int> layer_sizes {784, 800, 10};
  std::vector<Activation> activations {Activation::relu, Activation::softmax};

  MNISTLoader ldr_train, ldr_test;
  ldr_train.load("train image path", "train label path");
  ldr_test.load("test image path", "test label path");

  DNN dnn (layer_sizes, activations, ldr_train.getImageVec(), ldr_train.getLabelVec(), 10);
  dnn.trainADAM(0.2, 0.9, 0.999, 256, 1, 0.25, false, true, false, true);
  dnn.setDataset(ldr_test.getImageVec(), ldr_test.getLabelVec());
  dnn.test();

  // DNN dnn (
  //   "coefficients path",
  //   ldr_test.getImageVec(),
  //   ldr_test.getLabelVec(),
  //   10
  // );
  // dnn.setDataset(ldr_test.getImageVec(), ldr_test.getLabelVec());
  // dnn.test();
}

TEST_CASE("conversion") {
  float a = 1.082442;
  std::cout << "initial " << a << std::endl;
  char buf [4];

  uint32_t cast_data = std::bit_cast<uint32_t>(a);
  std::cout << cast_data << std::endl;

  buf[0] = static_cast<char>(0xff & (cast_data >> 24));
  buf[1] = static_cast<char>(0xff & (cast_data >> 16));
  buf[2] = static_cast<char>(0xff & (cast_data >> 8));
  buf[3] = static_cast<char>(0xff &  cast_data);

  std::cout << "buf 0: " << (uint32_t)(unsigned char)buf[0] << std::endl;
  std::cout << "buf 1: " << (uint32_t)(unsigned char)buf[1] << std::endl;
  std::cout << "buf 2: " << (uint32_t)(unsigned char)buf[2] << std::endl;
  std::cout << "buf 3: " << (uint32_t)(unsigned char)buf[3] << std::endl;

  uint32_t out_of_buf =
    (std::bit_cast<uint8_t>(buf[0]) << 24) |
    (std::bit_cast<uint8_t>(buf[1]) << 16) |
    (std::bit_cast<uint8_t>(buf[2]) << 8)  |
    (std::bit_cast<uint8_t>(buf[3]));

  std::cout << "buf0 out " << out_of_buf << std::endl;
  std::cout << "final " << std::bit_cast<float>(out_of_buf) << std::endl;
  REQUIRE ( std::bit_cast<float>(out_of_buf) == a );
}