#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "mnist_loader.h"

void MNISTLoader::load(
  std::string images_filepath,
  std::string labels_filepath) {
    char tmp [4] = {1, 1, 0, 0};
    // open images file
    std::ifstream images_file(images_filepath);

    if (!images_file.is_open()) throw std::runtime_error("Images file is not opened");
    images_file.read(tmp, 4);
    if(!(tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 8 && tmp[3] == 3)) throw std::runtime_error("Incorrect magic header for image dataset (should be 0 0 8 3)");

    // Resize images vector appropriately
    // 3 four byte integers store the dimensions of the images dataset
    // Dimension 0 is the size of the dataset, which is the variable part
    images_file.read(tmp, 4);
    int images_dataset_size = (0xff&tmp[0]) << 24 | (0xff&tmp[1]) << 16 | (0xff&tmp[2]) << 8 | (0xff&tmp[3]); // Bitshifts to guarantee compatibility
    images_.reserve(images_dataset_size*784);
    images_.resize(images_dataset_size*784);

    // Dimensions 1 and 2 are the size of each image, which are always 28x28
    // Ensure the rest of the dimensions are correct
    for (int i = 0; i < 2; i++) {
        images_file.read(tmp, 4);
        if(!(tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0 && tmp[3] == 28)) {
          throw std::runtime_error("Incorrect length or width dimension header for image dataset (should be 0 0 0 28)");
        }
    }

    // Load images
    char img_buf [28*28];
    for (int i = 0; i < images_dataset_size; i++) {
        images_file.read(img_buf, 28*28);
        for (int j = 0; j < 28*28; j++) {
            images_.at(28*28*i + j) = (float)(0xff&img_buf[j]) / 255.0;
        }
    }

    // close images file
    images_file.close();

    // open labels file
    std::ifstream labels_file(labels_filepath);

    if (!labels_file.is_open()) throw std::runtime_error("Labels file is not opened");
    labels_file.read(tmp, 4);
    if(!(tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 8 && tmp[3] == 1)) {
      throw std::runtime_error("Incorrect magic header for label dataset (should be 0 0 8 1)");
    }

    // Resize labels vector appropriately
    // 1 four byte integer stores the dimension of the labels dataset
    labels_file.read(tmp, 4);
    int labels_dataset_size = (0xff&tmp[0]) << 24 | (0xff&tmp[1]) << 16 | (0xff&tmp[2]) << 8 | (0xff&tmp[3]); // Bitshifts to guarantee compatibility
    labels_.reserve(labels_dataset_size);
    labels_.resize(labels_dataset_size);

    // Load labels
    char lbl_buf [labels_dataset_size];
    labels_file.read(lbl_buf, labels_dataset_size);
    for (int i = 0; i < labels_dataset_size; i++) {
        labels_[i] = (float)(0xff&lbl_buf[i]);
    }

    // close labels file
    labels_file.close();
}

std::vector<float>& MNISTLoader::getImageVec() {
    return images_;
}

std::vector<float>& MNISTLoader::getLabelVec() {
    return labels_;
}

int MNISTLoader::checkImageVecCapacity() {
    return images_.capacity();
}

int MNISTLoader::checkLabelVecCapacity() {
    return labels_.capacity();
}
