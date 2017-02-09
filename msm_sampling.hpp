#pragma once

#include <vector>
#include <string>

void
compute_prob_estimates(const std::vector<unsigned int>& pops
                     , const std::vector<std::vector<float>>& ref_coords
                     , float radius
                     , std::string fname_out);

void
sample_traj(const std::vector<unsigned int>& traj
          , const std::vector<unsigned int>& states
          , const std::vector<unsigned int>& pops
          , const std::vector<std::vector<float>>& coords
          , unsigned int batchsize
          , float radius
          , std::string fname_out);


