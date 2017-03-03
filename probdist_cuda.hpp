#pragma once

#define BSIZE 128

namespace CUDA {

  typedef std::unordered_map<unsigned int
                           , std::vector<float>> SplitFe;

  typedef std::unordered_map<unsigned int
                           , std::vector<std::vector<float>>> SplitCoord;


  struct GPUSettings {
    int id;
    std::vector<unsigned int> states;
    unsigned int n_dim;
    std::unordered_map<unsigned int, unsigned int> split_sizes;
    std::unordered_map<unsigned int, float*> fe;
    std::unordered_map<unsigned int, float*> coords;
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


  Pops
  calculate_populations_partial(const float* coords
                              , const std::vector<float>& sorted_coords
                              , const std::vector<float>& blimits
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu);
  
  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii);

} // end namespace CUDA

