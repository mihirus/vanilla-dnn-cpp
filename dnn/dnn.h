// #include <vector>
#include <Eigen/Dense>
#include <string>

enum class Activation {
  relu,
  softmax
};

class DNN {
public:

  DNN(
    std::vector<int> layer_sizes,
    std::vector<Activation> activations,
    std::vector<float>& images_ref,
    std::vector<float>& labels_ref,
    int labels_size
  );

  DNN(
    std::string filename,
    std::vector<float>& images_ref,
    std::vector<float>& labels_ref,
    int labels_size
  );

  void setDataset(
    std::vector<float>& images_ref,
    std::vector<float>& labels_ref
  );

  void trainADAM(
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
  );
  void test();

private:

  void process_sample(
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>* v_input_ptr,
    Eigen::Matrix<float, Eigen::Dynamic, 1>* v_input_noise_ptr,
    Eigen::Matrix<float, Eigen::Dynamic, 1>& true_value,
    bool inference_only
  );

  void activate(
    Eigen::Matrix<float, Eigen::Dynamic, 1>& input,
    Eigen::Matrix<float, Eigen::Dynamic, 1>& output,
    Activation activation
  );

  void initialize_matrix(
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>& m,
    int rows,
    int cols
  );

  void initialize_vector(
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>>& v,
    int rows
  );

  void step(float learning_rate, float beta1, float beta2, int t, int minibatch_size);

  void save();
  void load(std::string filename);

  // Can use any 4 byte data types
  template <typename T> void write4(T& data, char* buf);
  template <typename T> void read4(T& data, char* buf);

  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> m_weights_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> m_weight_grad_1st_moment_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> m_weight_grad_1st_moment_last_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> m_weight_grad_2nd_moment_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> m_weight_grad_2nd_moment_last_;

  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_biases_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_bias_grad_1st_moment_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_bias_grad_1st_moment_last_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_bias_grad_2nd_moment_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_bias_grad_2nd_moment_last_;


  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_neuron_inputs_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_neuron_input_grad_;

  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_neuron_outputs_;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> v_neuron_output_grad_;

  float loss_;
  Eigen::Matrix<float, Eigen::Dynamic, 1> v_inference_;

  std::vector<int> layer_sizes_;
  std::vector<Activation> activations_;
  std::vector<float>& images_ref_;
  std::vector<float>& labels_ref_;

  int labels_size_;
};