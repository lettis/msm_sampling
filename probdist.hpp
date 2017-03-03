#pragma once

#include <vector>
#include <list>
#include <functional>
#include <unordered_map>
#include <utility>

typedef std::pair<float, std::vector<float>> Sample;


float
fe_estimate(const std::vector<float>& xs
          , float dist2
          , const std::vector<float>& fe
          , const std::vector<std::vector<float>>& coords);


class StateSampler {
 public:
  StateSampler(const std::vector<unsigned int>& states
             , const std::vector<float>& fe
             , const std::vector<std::vector<float>>& ref_coords
             , float radius);
  Sample operator()(unsigned int state);

 private:
  // inits
  const std::vector<unsigned int>& _states;
  const std::vector<float>& _fe;
  const std::vector<std::vector<float>>& _ref_coords;
  float _radius;
  float _radius_squared;
  // bookkeeping
  unsigned int _n_frames_sampled;
  unsigned int _prev_state;
  Sample _prev_sample;
  // random number generator
  std::function<float()> _dice;
  // splitted reference values
  std::unordered_map<unsigned int, std::vector<float>> _fe_splitted;
  std::unordered_map<unsigned int, float> _max_fe_splitted;
  std::unordered_map<unsigned int, std::vector<std::pair<float,float>>>
    _min_max;
  std::unordered_map<unsigned int, std::vector<std::vector<float>>>
    _ref_coords_splitted;
  // dimensionality
  unsigned int _n_dim;
};

