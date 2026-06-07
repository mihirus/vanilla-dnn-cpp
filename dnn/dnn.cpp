#include "dnn.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>

DNN::DNN(
  std::vector<int> layer_sizes,
  std::vector<Activation> activations,
  std::vector<float>& images_ref,
  std::vector<float>& labels_ref,
  int labels_size
) :
  layer_sizes_ {layer_sizes},
  activations_ {activations},
  images_ref_ {images_ref},
  labels_ref_ {labels_ref},
  labels_size_{labels_size} {

  if (images_ref_.size() / layer_sizes.front() != labels_ref_.size()) throw std::runtime_error("Images dataset and labels dataset are not the same size");
  if (activations.size() != layer_sizes.size() - 1) throw std::runtime_error("Layer size list is not exactly one larger than activation list");

  float weights_inits_std_dev;
  std::default_random_engine generator;
  std::normal_distribution<float> distribution;

  for (int i = 0; i < layer_sizes.size()-1; i++) {

    // Layer input neurons
    // The very first input is a pointer to an external map
    if (i != 0) {
      initialize_vector(v_neuron_inputs_, layer_sizes.at(i));
      initialize_vector(v_neuron_input_grad_, layer_sizes.at(i));
    }

    // Layer output
    initialize_vector(v_neuron_outputs_, layer_sizes.at(i+1));
    initialize_vector(v_neuron_output_grad_, layer_sizes.at(i+1));

    // Layer weights
    initialize_matrix(m_weights_, layer_sizes.at(i+1), layer_sizes.at(i));

    // He initialization:
    // N~[0, 4/(dimension of input + dimension of output)]
    weights_inits_std_dev = std::sqrt(4.0 / (float)(layer_sizes.at(i+1)+ layer_sizes.at(i)));
    std::normal_distribution<float> distribution (0.0, weights_inits_std_dev);
    for (int row = 0; row < layer_sizes.at(i+1); row++) {
      for (int col = 0; col < layer_sizes.at(i); col++) {
        m_weights_.back()(row, col) = distribution(generator);
      }
    }

    // Layer weight gradients
    initialize_matrix(m_weight_grad_1st_moment_, layer_sizes.at(i+1), layer_sizes.at(i));
    initialize_matrix(m_weight_grad_1st_moment_last_, layer_sizes.at(i+1), layer_sizes.at(i));
    initialize_matrix(m_weight_grad_2nd_moment_, layer_sizes.at(i+1), layer_sizes.at(i));
    initialize_matrix(m_weight_grad_2nd_moment_last_, layer_sizes.at(i+1), layer_sizes.at(i));

    // Layer biases
    initialize_vector(v_biases_, layer_sizes.at(i+1));

    // Layer bias gradients
    initialize_vector(v_bias_grad_1st_moment_, layer_sizes.at(i+1));
    initialize_vector(v_bias_grad_1st_moment_last_, layer_sizes.at(i+1));
    initialize_vector(v_bias_grad_2nd_moment_, layer_sizes.at(i+1));
    initialize_vector(v_bias_grad_2nd_moment_last_, layer_sizes.at(i+1));
  }

  // Initialize inference
  v_inference_.resize(layer_sizes_.back(), 1);
  v_inference_.fill(0.0);
  
}

DNN::DNN(std::string filename, std::vector<float>& images_ref, std::vector<float>& labels_ref, int labels_size) :
  images_ref_ {images_ref},
  labels_ref_ {labels_ref},
  labels_size_ {labels_size} {
  load(filename); // Initializes layer_sizes_, m_weights_, v_biases_
}

void DNN::initialize_matrix(
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>& m,
  int rows,
  int cols
) {
  m.push_back(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(rows, cols));
  m.back().fill(0.0);
}

void DNN::initialize_vector(
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>>& v,
  int rows
) {
  v.push_back(Eigen::Matrix<float, Eigen::Dynamic, 1>(rows, 1));
  v.back().fill(0.0);
}

void DNN::setDataset(
  std::vector<float>& images_ref,
  std::vector<float>& labels_ref ) {

  if (images_ref_.size() / layer_sizes_.front() != labels_ref_.size()) throw std::runtime_error("Images dataset and labels dataset are not the same size");

  images_ref_ = images_ref;
  labels_ref_ = labels_ref;
}

// input_ptr -> output 0 -> input 0 -> ... -> input n-1 -> output n -> inference -> loss
void DNN::process_sample(
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>* v_input_ptr,
  Eigen::Matrix<float, Eigen::Dynamic, 1>* v_input_noise_ptr,
  Eigen::Matrix<float, Eigen::Dynamic, 1>& v_true_value,
  bool inference_only
) {

  if (v_input_ptr->rows() != layer_sizes_.front())
    throw std::runtime_error ("Attempted to pass an incorrectly sized input map");
  if (v_input_noise_ptr != nullptr && v_input_noise_ptr->rows() != layer_sizes_.front())
    throw std::runtime_error ("Attempted to pass an incorrectly sized input noise pointer map");
  if (v_true_value.rows() != labels_size_)
    throw std::runtime_error ("Attempted to pass an incorrectly sized true value");

  // FORWARD PROPAGATE

  for (int i = 0; i < layer_sizes_.size()-1; i++) {
    // Full connection
    if (i == 0 && v_input_noise_ptr != nullptr) {
      v_neuron_outputs_.front() = m_weights_.front() * ((*v_input_ptr) + (*v_input_noise_ptr)) + v_biases_.front();
    } else if (i == 0 && v_input_noise_ptr == nullptr) {
      v_neuron_outputs_.front() = m_weights_.front() * (*v_input_ptr) + v_biases_.front();
    } else {
      // Note that inputs are shifted by 1 bc the first input is an Eigen map (turns raw array into matrix)
      v_neuron_outputs_.at(i) = m_weights_.at(i) * v_neuron_inputs_.at(i-1) + v_biases_.at(i);
    }

    // Activation
    activate(
      v_neuron_outputs_.at(i),
      i < layer_sizes_.size()-2 ? v_neuron_inputs_.at(i) : v_inference_,
      activations_.at(i)
    );
  }

  // COMPUTE LOSS
  // Find max val
  float max_val = v_neuron_outputs_.back()(0);
  float denominator = 0.0;
  for (auto logit : v_neuron_outputs_.back()) {
    if (logit > max_val) max_val = logit;
  }

  // Subtract max val from all logits to prevent overflow
  for (auto logit : v_neuron_outputs_.back()) {
    denominator += std::exp(logit - max_val);
  }

  loss_ = 0.0;
  for (int i = 0; i < v_neuron_outputs_.back().rows(); i++) {
    // loss -= v_true_value(i) * std::log(v_output_(i));
    loss_ -= v_true_value(i) * (v_neuron_outputs_.back()(i) - max_val - std::log(denominator));
  }

  if (!inference_only) {
    // BACKPROPAGATE
    
    // First do inference to last output
    // Using softmax & ce loss so can use this simplification
    if (activations_.back() == Activation::softmax) {
      v_neuron_output_grad_.back() = v_inference_ - v_true_value;
    }

    // Then do the rest
    for (int i = layer_sizes_.size()-2; i > 0; i--) {
      // FC backprop
      m_weight_grad_1st_moment_.at(i).noalias() += v_neuron_output_grad_.at(i) * v_neuron_inputs_.at(i-1).transpose();
      v_bias_grad_1st_moment_.at(i).noalias() += v_neuron_output_grad_.at(i);
      v_neuron_input_grad_.at(i-1) = m_weights_.at(i).transpose() * v_neuron_output_grad_.at(i);

      // Activation backprop
      if (activations_.at(i-1) == Activation::relu) {
        for (int j = 0; j < v_neuron_inputs_.at(i-1).rows(); j++) {
          v_neuron_output_grad_.at(i-1)(j) = v_neuron_outputs_.at(i-1)(j) <= 0.0 ? 0.0 : v_neuron_input_grad_.at(i-1)(j);
        }
      }
    }

    // And finally first layer weights and biases
    m_weight_grad_1st_moment_.front().noalias() += v_neuron_output_grad_.front() * v_input_ptr->transpose();
    v_bias_grad_1st_moment_.front().noalias() += v_neuron_output_grad_.front();
  }
}

void DNN::activate(
  Eigen::Matrix<float, Eigen::Dynamic, 1>& input,
  Eigen::Matrix<float, Eigen::Dynamic, 1>& output,
  Activation activation) {

  assert (input.rows() == output.rows() && "activation function input size must equal output size");

  if (activation == Activation::relu) {
    for (int i = 0; i < input.rows(); i++) {
      // std::cout << i;
      output(i) = input(i) < 0.0 ? 0.0 : input(i);
    }
  } else if (activation == Activation::softmax) {
    float max_val = input(0);
    for (int i = 0; i < input.rows(); i++) {
      if (input(i) > max_val) max_val = input(i);
    }

    // Subtract maximum value from all exp(*)
    float denominator = 0.0;
    for (int i = 0; i < input.rows(); i++) {
      denominator += std::exp(input(i) - max_val);
    }

    // Outputs
    for (int i = 0; i < input.rows(); i++) {
      output(i) = std::exp(input(i) - max_val) / denominator;
    }
  }
}

void DNN::step(float learning_rate, float beta1, float beta2, int t, int minibatch_size) {

  for (int i = 0; i < layer_sizes_.size()-1; i++) {
    m_weight_grad_1st_moment_.at(i) = (beta1*m_weight_grad_1st_moment_last_.at(i))  + (1-beta1)*m_weight_grad_1st_moment_.at(i);
    v_bias_grad_1st_moment_.at(i) = (beta1*v_bias_grad_1st_moment_last_.at(i))  + (1-beta1)*v_bias_grad_1st_moment_.at(i);

    m_weight_grad_2nd_moment_.at(i) = (beta2*m_weight_grad_2nd_moment_last_.at(i))  + (1-beta2)*m_weight_grad_1st_moment_.at(i).array().square().matrix();
    v_bias_grad_2nd_moment_.at(i) = (beta2*v_bias_grad_2nd_moment_last_.at(i))  + (1-beta2)*v_bias_grad_1st_moment_.at(i).array().square().matrix();

    m_weights_.at(i).noalias() -= (learning_rate/(float)minibatch_size) * ((m_weight_grad_1st_moment_.at(i).array() / (1 - std::pow(beta1, t))) / ((m_weight_grad_2nd_moment_.at(i).array() / (1 - std::pow(beta2, t))).sqrt() + 1e-8)).matrix();
    v_biases_.at(i).noalias() -= (learning_rate/(float)minibatch_size) * ((v_bias_grad_1st_moment_.at(i).array() / (1 - std::pow(beta1, t))) / ((v_bias_grad_2nd_moment_.at(i).array() / (1 - std::pow(beta2, t))).sqrt() + 1e-8)).matrix();

    m_weight_grad_1st_moment_last_.at(i) = m_weight_grad_1st_moment_.at(i);
    v_bias_grad_1st_moment_last_.at(i) = v_bias_grad_1st_moment_.at(i);

    m_weight_grad_2nd_moment_last_.at(i) = m_weight_grad_2nd_moment_.at(i);
    v_bias_grad_2nd_moment_last_.at(i) = v_bias_grad_2nd_moment_.at(i);

    m_weight_grad_1st_moment_.at(i).fill(0.0);
    v_bias_grad_1st_moment_.at(i).fill(0.0);

    m_weight_grad_2nd_moment_.at(i).fill(0.0);
    v_bias_grad_2nd_moment_.at(i).fill(0.0);
  }
}

// Scheme:
// 1 byte, 0b10110100 - header
// 1 byte - number of layer sizes
// number of layer sizes * 4 bytes - layer sizes
// (number of layer sizes - 1) * 4 bytes - activations
// nth layer size * (n+1)th layer size - nth layer weights, column major
// (n+1)th layer size - nth layer biases
void DNN::save() {
  std::string filename = "mlp_coefficients_";

  // Layer sizes
  for (auto size : layer_sizes_) {
    filename += std::to_string(size) + "_";
  }

  // Local time
  std::chrono::current_zone()->to_local(std::chrono::system_clock::now());
  filename += std::format("{:%Y_%m_%d_%X}", std::chrono::current_zone()->to_local(std::chrono::system_clock::now()));

  // open file
  std::ofstream coefficients_file(filename);

  char buf [4];
  unsigned int coefficient;

  if (coefficients_file.is_open()) {
    // Write header
    coefficients_file.put((char)0b10110100);

    // Write number of sizes
    coefficients_file.put((char)layer_sizes_.size());

    // Write sizes
    for (int size : layer_sizes_) {
      write4(size, buf);
      coefficients_file.write(buf, 4);
    }

    // Write activations
    for (Activation act : activations_) {
      uint32_t act_int = static_cast<uint32_t>(act);
      write4(act_int, buf);
      coefficients_file.write(buf, 4);
    }

    // Write matrices
    for (int i = 0; i < layer_sizes_.size()-1; i++) {

      // Write weight coefficients
      for (int j = 0; j < layer_sizes_.at(i)*layer_sizes_.at(i+1); j++) {
        write4(*(m_weights_.at(i).data() + j), buf);
        coefficients_file.write(buf, 4);
      }
      std::cout << std::endl;

      // Write bias coefficients
      for (int j = 0; j < layer_sizes_.at(i+1); j++) {
        coefficient = std::bit_cast<uint32_t>(*(v_biases_.at(i).data() + j));
        write4(coefficient, buf);
        coefficients_file.write(buf, 4);
      }
    }
  }

  coefficients_file.close();
}

void DNN::load(std::string filename) {
  std::ifstream coefficients_file(filename);

  char buf [4];

  if (coefficients_file.is_open()) {
    coefficients_file.read(buf, 2);
    if ((unsigned char)buf[0] != 0xb4) throw std::runtime_error("Coefficients file does not have correct header 0xb4");

    layer_sizes_.resize((int)(unsigned char)buf[1]);
    activations_.resize((int)(unsigned char)buf[1] - 1);

    std::cout << "Layers resized to ";
    for (int i = 0; i < layer_sizes_.size(); i++) {
      coefficients_file.read(buf, 4);
      read4(layer_sizes_.at(i), buf);
      std::cout << layer_sizes_.at(i) << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < layer_sizes_.size()-1; i++) {
      coefficients_file.read(buf, 4);
      read4(activations_.at(i), buf);
    }

    for (int i = 0; i < layer_sizes_.size()-1; i++) {
      m_weights_.push_back(
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(layer_sizes_.at(i+1), layer_sizes_.at(i))
      );
      v_biases_.push_back(
        Eigen::Matrix<float, Eigen::Dynamic, 1>(layer_sizes_.at(i+1))
      );

      // Read weights
      for (int j = 0; j < layer_sizes_.at(i+1)*layer_sizes_.at(i); j++) {
        coefficients_file.read(buf, 4);
        read4(*(m_weights_.back().data() + j), buf);
      }

      // Read biases
      for (int j = 0; j < layer_sizes_.at(i+1); j++) {
        coefficients_file.read(buf, 4);
        read4(*(v_biases_.back().data() + j), buf);
      }

      // Input/output neurons
      if (i > 0) {
        v_neuron_inputs_.push_back(
          Eigen::Matrix<float, Eigen::Dynamic, 1>(layer_sizes_.at(i), 1)
        );
        v_neuron_inputs_.back().fill(0.0);
      }

      v_neuron_outputs_.push_back(
        Eigen::Matrix<float, Eigen::Dynamic, 1>(layer_sizes_.at(i+1), 1)
      );
      v_neuron_outputs_.back().fill(0.0);
    }

    // Initialize inference
    v_inference_.resize(layer_sizes_.back(), 1);
    v_inference_.fill(0.0);
  }

  // Check
  for (auto m : m_weights_) {
    std::cout << "Weights sized: " << m.rows() << " by " << m.cols() << std::endl;
  }
  for (auto v : v_biases_) {
    std::cout << "Biases sized: " << v.rows() << " by " << v.cols() << std::endl;
  }
  for (auto v : v_neuron_inputs_) {
    std::cout << "Layer input neurons sized: " << v.rows() << " by " << v.cols() << std::endl;
  }
  for (auto v : v_neuron_outputs_) {
    std::cout << "Layer output neurons sized: " << v.rows() << " by " << v.cols() << std::endl;
  }
}

template <typename T> void DNN::write4(T& data, char* buf) {
  uint32_t cast_data = std::bit_cast<uint32_t>(data);
  buf[0] = (char)(0xff & (cast_data >> 24));
  buf[1] = (char)(0xff & (cast_data >> 16));
  buf[2] = (char)(0xff & (cast_data >> 8));
  buf[3] = (char)(0xff &  cast_data);
}

template <typename T> void DNN::read4(T& data, char* buf) {
  uint32_t intermediate = std::bit_cast<uint8_t>(buf[0]) << 24 |
                          std::bit_cast<uint8_t>(buf[1]) << 16 |
                          std::bit_cast<uint8_t>(buf[2]) << 8  |
                          std::bit_cast<uint8_t>(buf[3]);
  data = std::bit_cast<T>(intermediate);
}

// Layer 1 input map connected in here
void DNN::trainADAM(
  float learning_rate,
  float beta1,
  float beta2,
  int minibatch_size,
  int num_epochs,
  float input_noise_std,
  bool minibatch_prints,
  bool epoch_prints,
  bool timing_prints,
  bool save_coefficients
) {

  const int dataset_size = images_ref_.size() / layer_sizes_.front();
  float minibatch_loss_avg = 0.0;
  float minibatch_accuracy = 0.0;
  float epoch_loss_avg = 0.0;
  float epoch_accuracy = 0.0;

  int minibatch_i;
  int true_val_ind;

  // Timing
  std::chrono::time_point<std::chrono::steady_clock> t1, t2;

  // Max coeff locations
  Eigen::Index max_row, max_col;

  // Generate array of consecutive numbers
  std::vector<int> dataset_indices (dataset_size);
  for (int i = 0; i < dataset_size; i++) {
    dataset_indices[i] = i;
  }

  // PRNG for shuffling every epoch
  std::random_device rd; // seed
  std::mt19937 g(rd());

  // True value array
  Eigen::Matrix<float, Eigen::Dynamic, 1> true_value (labels_size_, 1);

  // Random noise added to image

  std::normal_distribution<float> distribution(0.0, input_noise_std);
  Eigen::Matrix<float, Eigen::Dynamic, 1> v_input_noise (layer_sizes_.front(), 1);
  for (int i = 0; i < layer_sizes_.front(); i++) {
    v_input_noise(i) = distribution(g);
  }

  for (int epoch = 0; epoch < num_epochs; epoch++) {

    // Shuffle dataset indices
    std::shuffle(dataset_indices.begin(), dataset_indices.end(), g);

    // Work through entire dataset
    for (int i = 0; i < dataset_size; i++) {

      // Create network input map
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> v_input (
        images_ref_.data() + (layer_sizes_.front() * dataset_indices.at(i)),
        layer_sizes_.front(),
        1
      );

      // Reset true value
      true_value.fill(0.0);
      true_val_ind = (int)labels_ref_.at(dataset_indices.at(i));
      true_value(true_val_ind) = 1.0;

      // Update gradients
      t1 = std::chrono::steady_clock::now();
      process_sample(&v_input, &v_input_noise, true_value, false);
      t2 = std::chrono::steady_clock::now();

      // Determine the selected class
      v_inference_.maxCoeff(&max_row, &max_col);

      if (minibatch_prints) {
        minibatch_i = i%minibatch_size + 1;
        minibatch_loss_avg = (minibatch_loss_avg*(minibatch_i-1) + loss_) / (float)(minibatch_i);
        minibatch_accuracy = (minibatch_accuracy*(minibatch_i-1) + ((int)max_row == true_val_ind ? 1.0 : 0.0)) / (float)(minibatch_i);
      }

      if (epoch_prints) {
        epoch_loss_avg = (epoch_loss_avg*i + loss_) / (float)(i+1);
        epoch_accuracy = (epoch_accuracy*i + ((int)max_row == true_val_ind ? 1.0 : 0.0)) / (float)(i+1);
      }

      if ((i+1) % minibatch_size == 0) {
        // step
        step(learning_rate, beta1, beta2, (i+1)/minibatch_size, minibatch_size);

        if (timing_prints) std::cout << "Gradient update duration (us): " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;

        if (minibatch_prints) {
          std::cout << "Epoch " << epoch << " minibatch " << i/minibatch_size << std::endl <<
            " Minibatch loss average: " << minibatch_loss_avg << std::endl <<
            " Minibatch accuracy: " << minibatch_accuracy << std::endl << std::endl;
        }

        // Generate new input noise each minibatch
        for (int i = 0; i < layer_sizes_.front(); i++) {
          v_input_noise(i) = distribution(g);
        }
      }
    }

    if (epoch_prints) {
      std::cout << "Epoch " << epoch << std::endl <<
        " Epoch loss average: " << epoch_loss_avg << std::endl <<
        " Epoch accuracy: " << epoch_accuracy << std::endl << std::endl;
    }
  }

  if (save_coefficients) {
    save();
  }
}

void DNN::test() {
  const int dataset_size = images_ref_.size() / layer_sizes_.front();
  float loss_avg = 0.0;
  float accuracy = 0.0;

  int true_val_ind;

  // Max coeff locations
  Eigen::Index max_row, max_col;

  // True value array
  Eigen::Matrix<float, Eigen::Dynamic, 1> true_value (layer_sizes_.back(), 1);

  // Work through entire dataset
  for (int i = 0; i < dataset_size; i++) {

    // Create network input map
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> v_input (
      images_ref_.data() + (layer_sizes_.front() * i),
      layer_sizes_.front(),
      1
    );

    // Reset true value
    true_value.fill(0.0);
    true_val_ind = (int)labels_ref_.at(i);
    true_value(true_val_ind) = 1.0;

    // Inference
    process_sample(&v_input, nullptr, true_value, true);

    // Determine the selected class
    v_inference_.maxCoeff(&max_row, &max_col);

    loss_avg = (loss_avg*i + loss_) / (float)(i+1);
    accuracy = (accuracy*i + ((int)max_row == true_val_ind ? 1.0 : 0.0)) / (float)(i+1);
  }

  std::cout << "Test loss: " << loss_avg << " Accuracy: " << accuracy << std::endl;

}