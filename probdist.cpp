
#include <vector>
#include <utility>
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <omp.h>

#include "probdist.hpp"
#include "tools.hpp"


float
fe_estimate(const std::vector<float>& xs
          , float dist2
          , const std::vector<float>& fe
          , const std::vector<std::vector<float>>& coords) {
  unsigned int nrow = coords.size();
  unsigned int ncol = xs.size();
  float d, d2;
  unsigned int i,j;
  double est_fe = 0;
  unsigned int n_neighbors = 0;
  #pragma omp parallel for\
    default(none)\
    private(i,j,d,d2)\
    firstprivate(dist2,ncol,nrow)\
    shared(fe,xs,coords,n_neighbors)\
    reduction(+:est_fe)
  for (i=0; i < nrow; ++i) {
    d2 = 0;
    for (j=0; j < ncol; ++j) {
      d = coords[i][j] - xs[j];
      d2 += d*d;
    }
    if (d2 < dist2) {
      #pragma omp atomic
      est_fe += fe[i];
      #pragma omp atomic
      ++n_neighbors;
    }
  }
  // if apart from others, use max. free energy
  if (n_neighbors == 0) {
    est_fe = (*std::max_element(fe.begin()
                              , fe.end()));
  } else {
    est_fe = (est_fe / n_neighbors);
  }
  return est_fe;
}


StateSampler::StateSampler(const std::vector<unsigned int>& states
                         , const std::vector<float>& fe
                         , const std::vector<std::vector<float>>& ref_coords
                         , float radius)
  : _states(states)
  , _fe(fe)
  , _ref_coords(ref_coords)
  , _radius(radius)
  , _radius_squared(radius*radius)
  , _n_frames_sampled(0) {
  // initialize random number generator
  _dice = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                  , std::mt19937(std::random_device()()));
  // split probs / coords into states
  std::unordered_set<unsigned int> state_names(states.begin()
                                             , states.end());
  // init splitted fe/coords and store
  for (unsigned int s: state_names) {
    _fe_splitted[s] = {};
    _ref_coords_splitted[s] = {};
  }
  for (unsigned int i=0; i < fe.size(); ++i){
    _fe_splitted[states[i]].push_back(fe[i]);
    _ref_coords_splitted[states[i]].push_back(ref_coords[i]);
  }
  // minima/maxima of ref.-coords. for sampling
  for (unsigned int s: state_names) {
    _min_max[s] = col_min_max(_ref_coords_splitted[s]);
    _max_fe_splitted[s] = (*std::max_element(_fe_splitted[s].begin()
                                           , _fe_splitted[s].end()));
  }
  _n_dim = ref_coords[0].size();
}


//TODO: idea: use simulation temp + ref. temp to scale according to temperature
//TODO: step-scaling per dimension to control autocorrelation


Sample
StateSampler::operator()(unsigned int state) {
  std::vector<float> new_sample_coords(_n_dim);
  float new_sample_fe;
  auto new_fe = [&]() -> float {
    return fe_estimate(new_sample_coords
                     , _radius_squared
                     , _fe_splitted[state]
                     , _ref_coords_splitted[state]);
  };
  if (state == _prev_state) {
    //TODO step width scaling per dimension as input parameter
    float step_width = _radius;
    bool no_sample_found = true;
    while (no_sample_found) {
      for (unsigned int i=0; i < _n_dim; ++i) {
        // update coords:
        //   new = old + uniform_sample([-step,step])
        new_sample_coords[i] = _prev_sample.second[i]
                             + _dice()*2*step_width
                             - step_width;
      }
      // Metropolis sampling:
      //   accept coords if smaller energy than previous
      //   or if Boltzmann weight of energy-difference is higher
      //   than random value.
      new_sample_fe = new_fe();
      if (new_sample_fe < _max_fe_splitted[state]) {
        if (new_sample_fe < _prev_sample.first
         || _dice() < std::min(1.0f
                             , std::exp(_prev_sample.first - new_sample_fe))) {
          no_sample_found = false;
        }
      }
    }
  } else {
    bool no_sample_found = true;
    while (no_sample_found) {
      // sample new coordinates 'out of the blue'
      for (unsigned int i=0; i < _n_dim; ++i) {
        new_sample_coords[i] = _min_max[state][i].first
                             + _dice()
                             * (_min_max[state][i].second
                              - _min_max[state][i].first);
      }
      // always accept new sample as long as energy < max(state_energy)
      new_sample_fe = new_fe();
      if (new_sample_fe < _max_fe_splitted[state]) {
        no_sample_found = false;
      }
    }
  }
  // bookkeeping for Metropolis algorithm
  ++_n_frames_sampled;
  Sample new_sample = {new_sample_fe
                     , new_sample_coords};
  _prev_sample = new_sample;
  _prev_state = state;
  return new_sample;
}

