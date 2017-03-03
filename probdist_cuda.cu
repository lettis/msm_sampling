
namespace CUDA {
  void
  check_error(std::string msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: "
                << msg << "\n"
                << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int
  get_num_gpus() {
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    check_error("trying to get number of available GPUs");
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found."
                << std::endl
                << "       if you are sure to have one,"
                << std::endl
                << "       check your device drivers!"
                << std::endl;
      exit(EXIT_FAILURE);
    } else {
      return n_gpus;
    }
  }

  GPUSettings
  prepare_gpu(int i_gpu
            , unsigned int n_dim
            , std::vector<unsigned int> states
            , const SplitFe& fe
            , const SplitCoord& ref_coords) {
    GPUSettings gpu;
    gpu.id = i_gpu;
    gpu.n_dim = n_dim;
    gpu.states = states;
    cudaSetDevice(i_gpu);
    check_error("setting CUDA device");
    for (unsigned int state: states) {
      unsigned int split_size = fe[state].size();
      gpu.split_sizes[state] = split_size;
      //// reserve memory
      cudaMalloc((void**) &gpu.fe[state]
               , sizeof(float) * split_size);
      cudaMalloc((void**) &gpu.coords[state]
               , sizeof(float) * split_size * gpu.n_dim);
      check_error("device memory allocation");
      //// copy data to device
      cudaMemcpy(gpu.fe[state]
               , fe[state].data()
               , sizeof(float) * split_size
               , cudaMemcpyHostToDevice);
      check_error("copying of state-splitted free energies");
      // reference coords in 1D array (row-major order)
      std::vector<float> tmp_coords(n_dim * split_size);
      for (unsigned int i=0; i < split_size; ++i) {
        for (unsigned int j=0; j < n_dim; ++j) {
          tmp_coords[i*split_size+j];
        }
      }
      cudaMemcpy(gpu.coords[state]
               , tmp_coords.data()
               , sizeof(float) * n_dim * split_size
               , cudaMemcpyHostToDevice);
      check_error("copying of state-splitted coordinates");
    }
    return gpu;
  }

  void
  clear_gpu(GPUSettings gpu) {
    cudaSetDevice(gpu.id);
    check_error("setting CUDA device");
    for (unsigned int state: gpu.states) {
      cudaFree(gpu.fe[state]);
      check_error("freeing memory for free energies");
      cudaFree(gpu.coords[state]);
      check_error("freeing memory for coordinates");
    }
  }


  


  //TODO reuse for FE estimate
/*
  Pops
  calculate_populations_per_gpu(const float* coords
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu) {
    using Clustering::Tools::min_multiplicator;
    unsigned int n_radii = radii.size();
    std::vector<float> rad2(n_radii);
    for (std::size_t i=0; i < n_radii; ++i) {
      rad2[i] = radii[i]*radii[i];
    }
    // GPU setup
    cudaSetDevice(i_gpu);
    float* d_coords;
    float* d_rad2;
    unsigned int* d_pops;
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMalloc((void**) &d_pops
             , sizeof(unsigned int) * n_rows * n_radii);
    cudaMalloc((void**) &d_rad2
             , sizeof(float) * n_radii);
    check_error("pop-calc device mallocs");
    cudaMemset(d_pops
             , 0
             , sizeof(unsigned int) * n_rows * n_radii);
    check_error("pop-calc memset");
    cudaMemcpy(d_coords
             , coords
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    cudaMemcpy(d_rad2
             , rad2.data()
             , sizeof(float) * n_radii
             , cudaMemcpyHostToDevice);
    check_error("pop-calc mem copies");
    int max_shared_mem;
    cudaDeviceGetAttribute(&max_shared_mem
                         , cudaDevAttrMaxSharedMemoryPerBlock
                         , i_gpu);
    check_error("getting max shared mem size");
    unsigned int block_size = BSIZE_POPS;
    unsigned int shared_mem = 2 * block_size * n_cols * sizeof(float);
    if (shared_mem > max_shared_mem) {
      std::cerr << "error: max. shared mem per block too small on this GPU.\n"
                << "       either reduce BSIZE_POPS or get a better GPU."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int block_rng = min_multiplicator(i_to-i_from, block_size);
    Clustering::logger(std::cout) << "# blocks needed: "
                                  << block_rng << std::endl;
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      Clustering::Density::CUDA::Kernel::population_count
      <<< block_rng
        , block_size
        , shared_mem >>> (i*block_size
                        , d_coords
                        , n_rows
                        , n_cols
                        , d_rad2
                        , n_radii
                        , d_pops
                        , i_from
                        , i_to);
    }
    cudaDeviceSynchronize();
    check_error("after kernel loop");
    // get partial results from GPU
    std::vector<unsigned int> partial_pops(n_rows*n_radii);
    cudaMemcpy(partial_pops.data()
             , d_pops
             , sizeof(unsigned int) * n_rows * n_radii
             , cudaMemcpyDeviceToHost);
    // sort into resulting pops
    Pops pops;
    for (unsigned int r=0; r < n_radii; ++r) {
      pops[radii[r]].resize(n_rows, 0);
      for (unsigned int i=i_from; i < i_to; ++i) {
        pops[radii[r]][i] = partial_pops[r*n_rows+i];
      }
    }
    cudaFree(d_coords);
    cudaFree(d_rad2);
    cudaFree(d_pops);
    return pops;
  }
*/


  std::pair<unsigned int, float>
  fe_estimate_partial(const std::vector<float>& xs
                    , unsigned int state
                    , unsigned int i_from
                    , unsigned int i_to
                    , GPUSettings& gpu) {
    //TODO
  }

  float
  fe_estimate(const std::vector<float>& xs
            , unsigned int state
            , const std::vector<GPUSettings>& gpus) {
    int n_gpus = gpus.size();
    if (n_gpus == 0) {
      std::cerr << "error: unable to estimate free energies on GPU(s)."
                << std::endl
                << "       no GPUs have been provided."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int n_rows = gpus[0].split_sizes[state];
    unsigned int gpu_range = n_rows / n_gpus;
    int i;
    // partial estimates: pair of {#neighbors, sum(FE)}
    std::vector<std::pair<unsigned int, float>> partial_estimates(n_gpus);
    #pragma omp parallel for default(none)\
      private(i)\
      firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
      shared(partial_pops,radii,coords)\
      num_threads(n_gpus)\
      schedule(dynamic,1)
    for (i=0; i < n_gpus; ++i) {
      // use available GPUs in parallel to
      // compute partial estimates
      partial_estimates[i] = fe_estimate_partial(xs
                                               , state
                                               , i*gpu_range
                                               , i == (n_gpus-1)
                                                   ? n_rows
                                                   : (i+1)*gpu_range
                                               , gpus[i]);
    }
    // combine results
    unsigned int n_neighbors = 0;
    float fe_estimate = 0.0f;
    for (auto result: partial_estimates) {
      n_neighbors += result.first;
      fe_estimate += result.second;
    }
    return fe_estimate / ((float) n_neighbors);
  }

}

