#include "mnist_loader.h"
#include <iostream>
#include <catch2/catch_test_macros.hpp>

std::string train_imgs;
std::string train_lbls;
std::string test_imgs;
std::string test_lbls;

bool tol(float a, float b, float tol) {
  return (a-b < tol || b-a < tol);
}

TEST_CASE("MNISTLoader") {
    MNISTLoader ldr;
    std::cout << train_imgs << std::endl;

    SECTION( "Test incorrect loading" ) {
      try {
        ldr.load("", "");
      } catch (std::runtime_error& e) {
        std::cerr << "Error successfully induced: " << e.what() << std::endl;
      }
    }

    SECTION( "Ensure loading is correct for MNIST test set" ) {
        REQUIRE ( ldr.checkImageVecCapacity() == 0 );
        REQUIRE ( ldr.checkLabelVecCapacity() == 0 );
        ldr.load(test_imgs, test_lbls);
        REQUIRE ( ldr.checkImageVecCapacity() == 10000*784 );
        REQUIRE ( ldr.checkLabelVecCapacity() == 10000 );
        REQUIRE ( tol(ldr.getImageVec().at(784*1 + (5*16 + 14)), 116.0, 1e-12) );
        REQUIRE ( tol(ldr.getImageVec().at(784*1 + (5*16 + 15)), 125.0, 1e-12) );
        REQUIRE ( ldr.getLabelVec()[0] == 7);
        REQUIRE ( ldr.getLabelVec()[5] == 1);
    }

    SECTION( "Ensure loading is correct for MNIST training set" ) {
        REQUIRE ( ldr.checkImageVecCapacity() == 0 );
        REQUIRE ( ldr.checkLabelVecCapacity() == 0 );
        ldr.load(train_imgs, train_lbls);
        REQUIRE ( ldr.checkImageVecCapacity() == 60000*784 );
        REQUIRE ( ldr.checkLabelVecCapacity() == 60000 );
        REQUIRE ( tol(ldr.getImageVec().at(784*0 + (12*16 + 10)), 0.0, 1e-12) );
        REQUIRE ( tol(ldr.getImageVec().at(784*0 + (12*16 + 11)), 49.0, 1e-12) );
        REQUIRE ( ldr.getLabelVec()[0] == 5);
        REQUIRE ( ldr.getLabelVec()[5] == 2);
    }
}