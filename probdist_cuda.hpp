#pragma once

#define BSIZE 128

#include <unordered_map>
#include <vector>


namespace CUDA {

  typedef std::unordered_map<unsigned int
                           , std::vector<float>> SplitFe;

  typedef std::unordered_map<unsigned int
                           , std::vector<std::vector<float>>> SplitCoord;

  struct GPUSettings {
    int id;
    std::vector<unsigned int> states;
    unsigned int n_dim;
    float* xs;
    std::unordered_map<unsigned int, unsigned int> split_sizes;
    std::unordered_map<unsigned int, float*> fe;
    std::unordered_map<unsigned int, float*> coords;
    float* est_fe;
    unsigned int* est_neighbors;
  };

  void
  check_error(std::string msg="");

  int
  get_num_gpus();

  GPUSettings
  prepare_gpu(int i_gpu
            , unsigned int n_dim
            , std::vector<unsigned int> states
            , const SplitFe& fe
            , const SplitCoord& ref_coords);

  void
  clear_gpu(GPUSettings settings);

  //! return minimum multiplicator to fulfill result * mult >= orig
  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult);

  float
  fe_estimate(const std::vector<float>& xs
            , float rad2
            , unsigned int state
            , const std::vector<GPUSettings>& gpus);

} // end namespace CUDA

