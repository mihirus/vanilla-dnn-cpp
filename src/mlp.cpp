#include "dnn.h"
#include "mnist_loader.h"

#include <iostream>
#include <string>
#include <filesystem>

/**
 * Format:
 * train will autosave labels into a binary named mlp_coefficients_<datetime of start run>
 * activations set to relu layer and softmax output
 * 
  --train <train images path> <train labels path> <test images path> <test labels path> --layer_sizes <sizes>
  --load <test images path> <test labels path> <coefficients path>

 */

int main (int argc, char *argv[]) {
  if (argc < 5) {
    return 0;
  }

  bool train = false;
  bool load = false;
  bool coefficients = false;
  std::string train_images_path;
  std::string train_labels_path;
  std::string test_images_path;
  std::string test_labels_path;
  std::string coefficients_path;
  std::vector<int> layer_sizes;
  std::vector<Activation> activations;

  for (int i = 0; i < argc-1; i++) {
    if (std::string_view(argv[i]) == "--train") {
      train = true;
      train_images_path = std::filesystem::current_path().append(argv[i+1]);
      train_labels_path = std::filesystem::current_path().append(argv[i+2]);
      std::cout << train_images_path << std::endl;
      test_images_path = std::filesystem::current_path().append(argv[i+3]);
      test_labels_path = std::filesystem::current_path().append(argv[i+4]);
      i+=4;
    } else if (std::string_view(argv[i]) == "--load") {
      load = true;
      test_images_path = std::filesystem::current_path().append(argv[i+1]);
      test_labels_path = std::filesystem::current_path().append(argv[i+2]);
      coefficients_path = std::filesystem::current_path().append(argv[i+3]);
      i+=3;
    } else if (std::string_view(argv[i]) == "--layer_sizes") {
      for (int j = i+1; j < argc; j++) {
        if (std::string_view(argv[j]).substr(0,1) != "-") {
          layer_sizes.push_back(std::atoi(argv[j]));
        } else {
          i = j;
          break;
        }
      }
    }
  }

  if (train) {
    // Create activations
    for (int i = 0; i < layer_sizes.size()-2; i++){
      activations.push_back(Activation::relu);
    }
    activations.push_back(Activation::softmax);
  }
  std::cout << std::endl;

  MNISTLoader ldr_test;
  ldr_test.load(test_images_path, test_labels_path);
  if (train && !load) {
    std::cout << "training" << std::endl;
    MNISTLoader ldr_train;
    ldr_train.load(train_images_path, train_labels_path);
    DNN dnn (layer_sizes, activations, ldr_train.getImageVec(), ldr_train.getLabelVec(), 10);
    dnn.trainADAM(0.2, 0.9, 0.999, 256, 100, 0.25, false, true, false, true);
    std::cout << "testing" << std::endl;
    dnn.setDataset(ldr_test.getImageVec(), ldr_test.getLabelVec());
    dnn.test();
  } else if (load && !train) {
    DNN dnn (coefficients_path, ldr_test.getImageVec(), ldr_test.getLabelVec(), 10);
    std::cout << "testing" << std::endl;
    dnn.test();
  }
  return 0;
}