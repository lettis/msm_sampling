
#include "probdist_cuda.hpp"

#include <limits>
#include <iostream>

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
      std::cerr << "error: no CUDA-compatible GPU(s) found."
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
    //// reserve memory for reference point (aka 'xs')
    cudaMalloc((void**) &gpu.xs
             , sizeof(float) * n_dim);
    check_error("malloc xs");
    unsigned int max_split_size = 0;
    for (unsigned int state: states) {
      unsigned int split_size = fe.at(state).size();
      max_split_size = std::max(split_size
                              , max_split_size);
      gpu.split_sizes[state] = split_size;
      //// reserve memory for references
      cudaMalloc((void**) &gpu.fe[state]
               , sizeof(float) * split_size);
      check_error("malloc reference FE");
      cudaMalloc((void**) &gpu.coords[state]
               , sizeof(float) * split_size * gpu.n_dim);
      check_error("malloc reference coords");
      //// copy data to device
      cudaMemcpy(gpu.fe[state]
               , fe.at(state).data()
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
    //// allocate memory for partial results
    unsigned int min_result_size = max_split_size/32 + 1;
    cudaMalloc((void**) gpu.est_fe
             , sizeof(float) * min_result_size);
    check_error("malloc partial fe results");
    cudaMalloc((void**) gpu.est_neighbors
             , sizeof(unsigned int) * min_result_size);
    check_error("malloc partial neighbor count");
    // ... and return GPU-settings
    return gpu;
  }

  void
  clear_gpu(GPUSettings gpu) {
    cudaSetDevice(gpu.id);
    check_error("setting CUDA device");
    cudaFree(gpu.xs);
    check_error("freeing memory for xs");
    for (unsigned int state: gpu.states) {
      cudaFree(gpu.fe[state]);
      check_error("freeing memory for free energies");
      cudaFree(gpu.coords[state]);
      check_error("freeing memory for coordinates");
    }
    cudaFree(gpu.est_fe);
    check_error("freeing memory for partial fe results");
    cudaFree(gpu.est_neighbors);
    check_error("freeing memory for partial neighbor count");
  }

  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult) {
    return (unsigned int) std::ceil(orig / ((float) mult));
  };

  __global__ void
  fe_estimate_krnl(float* xs
                 , float* ref_coords
                 , float* ref_fe
                 , float rad2
                 , unsigned int* est_neighbors
                 , float* est_fe
                 , unsigned int n_rows
                 , unsigned int n_cols
                 , unsigned int i_from
                 , unsigned int i_to) {
    // CUDA-specific indices for block, thread and global
    unsigned int bsize = blockDim.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // locally shared memory for fast access
    //  coords has size [(bsize+1) x n_cols] for
    //  bsize rows of ref-coords + 1 row of xs.
    extern __shared__ float s_coords[];
    __shared__ float s_fe[BSIZE];
    __shared__ unsigned int s_neighbors[BSIZE];
    if (gid < i_to) {
      // coalasced read of reference coords/fe
      for (unsigned int j=0; j < n_cols; ++j) {
        s_coords[(tid+1)*n_cols + j] = ref_coords[(tid+bsize*bid)*n_cols+j];
      }
      s_fe[tid] = ref_fe[gid];
      s_neighbors[tid] = 1;
      if (tid == 0) {
        // read xs to local mem
        for (unsigned int j=0; j < n_cols; ++j) {
          s_coords[j] = xs[j];
        }
      }
    } else {
      s_fe[tid] = 0.0f;
      s_neighbors[tid] = 0;
    }
    __syncthreads();
    // filter out all frames with distance (d2)
    // larger than cutoff (rad2)
    if (gid < i_to) {
      float d2 = 0.0f;
      for (unsigned int j=0; j < n_cols; ++j) {
        float d = (s_coords[(tid+1)*n_cols + j] - xs[j]);
        d2 += d*d;
      }
      if (rad2 < d2) {
        s_fe[tid] = 0.0f;
        s_neighbors[tid] = 0;
      }
    }
    __syncthreads();
    // local reduction of intermediate results
    // with unrolled loop in single warp
    for (unsigned int s=bsize/2; s > 32; s>>=1) {
      if (tid < s) {
        s_fe[tid] += s_fe[tid+s];
        s_neighbors[tid] += s_neighbors[tid+s];
      }
      __syncthreads();
    }
    // 32 is common warp size for all CUDA devices
    if (tid < 32) {
      // threads inside single warp are implicitly synced
      // => no __syncthreads() call necessary
      s_fe[tid] += s_fe[tid+32];
      s_neighbors[tid] += s_neighbors[tid+32];
      s_fe[tid] += s_fe[tid+16];
      s_neighbors[tid] += s_neighbors[tid+16];
      s_fe[tid] += s_fe[tid+ 8];
      s_neighbors[tid] += s_neighbors[tid+ 8];
      s_fe[tid] += s_fe[tid+ 4];
      s_neighbors[tid] += s_neighbors[tid+ 4];
      s_fe[tid] += s_fe[tid+ 2];
      s_neighbors[tid] += s_neighbors[tid+ 2];
      s_fe[tid] += s_fe[tid+ 1];
      s_neighbors[tid] += s_neighbors[tid+ 1];
    }
    // write results to global memory
    if (tid == 0) {
      est_fe[bid] = s_fe[0];
      est_neighbors[bid] = s_neighbors[0];
    }
  }



  float
  fe_estimate(const std::vector<float>& xs
            , float rad2
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
    unsigned int n_rows = gpus[0].split_sizes.at(state);
    unsigned int n_cols = gpus[0].n_dim;
    unsigned int gpu_range = n_rows / n_gpus;
    int i_gpu;
    unsigned int i_from;
    unsigned int i_to;
    unsigned int block_rng;
    unsigned int shared_mem_size;
    // partial estimates: pair of {#neighbors, sum(FE)}
    std::vector<std::pair<unsigned int, float>> partial_estimates(n_gpus);
    //// parallelize over available GPUs
    #pragma omp parallel for default(none)\
      private(i_gpu,i_from,i_to,block_rng,shared_mem_size)\
      firstprivate(rad2,n_gpus,n_rows,n_cols,gpu_range)\
      shared(partial_estimates,state,xs,gpus)\
      num_threads(n_gpus)\
      schedule(dynamic,1)
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaSetDevice(i_gpu);
      check_error("set device");
      // set ranges for this GPU
      i_from = i_gpu * gpu_range;
      if (i_gpu == n_gpus-1) {
        i_to = n_rows;
      } else {
        i_to = (i_gpu+1) * gpu_range;
      }
      cudaMemcpy(gpus[i_gpu].xs
               , xs.data()
               , sizeof(float) * n_cols
               , cudaMemcpyHostToDevice);
      check_error("copy reference point coordinates to device");
      //TODO implement min_multiplicator
      block_rng = min_multiplicator(i_to-i_from, BSIZE);
      shared_mem_size = sizeof(float) * (BSIZE+1) * n_cols;
      // kernel call
      fe_estimate_krnl
      <<< block_rng
        , BSIZE
        , shared_mem_size >>> (gpus[i_gpu].xs
                             , gpus[i_gpu].coords.at(state)
                             , gpus[i_gpu].fe.at(state)
                             , rad2
                             , gpus[i_gpu].est_neighbors
                             , gpus[i_gpu].est_fe
                             , n_rows
                             , n_cols
                             , i_from
                             , i_to);
      cudaDeviceSynchronize();
      check_error("after kernel call");
      // retrieve partial results
      std::vector<float> est_fe(block_rng);
      std::vector<unsigned int> est_neighbors(block_rng);
      cudaMemcpy(gpus[i_gpu].est_fe
               , est_fe.data()
               , sizeof(float) * block_rng
               , cudaMemcpyDeviceToHost);
      check_error("copy fe estimate from device");
      cudaMemcpy(gpus[i_gpu].est_neighbors
               , est_neighbors.data()
               , sizeof(float) * block_rng
               , cudaMemcpyDeviceToHost);
      check_error("copy neighbor estimate from device");
      // accumulate partial results per GPU
      for (auto fe: est_fe) {
        partial_estimates[i_gpu].second += fe;
      }
      for (auto n_neighbors: est_neighbors) {
        partial_estimates[i_gpu].first += n_neighbors;
      }
    }
    // combine results from GPUs
    unsigned int n_neighbors = 0;
    float fe_estimate = 0.0f;
    for (auto result: partial_estimates) {
      n_neighbors += result.first;
      fe_estimate += result.second;
    }
    if (n_neighbors == 0) {
      fe_estimate = std::numeric_limits<float>::max();
    } else {
      fe_estimate /= ((float) n_neighbors);
    }
    // final free energy estimate
    return fe_estimate;
  }

} // end namespace CUDA

