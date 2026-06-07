#include <vector>
#include <array>
#include <string>

class MNISTLoader {
public:
    void load(std::string images_filepath, std::string labels_filepath);
    std::vector<float>& getImageVec();
    std::vector<float>& getLabelVec();

    int checkImageVecCapacity();
    int checkLabelVecCapacity();

private:
    std::vector<float> images_;
    std::vector<float> labels_;
};
