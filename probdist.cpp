
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


double
sum_probs(const std::list<Sample>& samples) {
  double sum = 0.0;
  for (Sample s: samples) {
    sum += s.first;
  }
  return sum;
}

double
prob_estimate(const std::vector<float>& xs
            , float dist2
            , const std::vector<double>& probs
            , const std::vector<std::vector<float>>& coords) {
  unsigned int nrow = coords.size();
  unsigned int ncol = xs.size();
  float d, d2;
  unsigned int i,j;
  double p=0;

  #pragma omp parallel for\
    default(none)\
    private(i,j,d,d2)\
    firstprivate(dist2,ncol,nrow)\
    shared(probs,xs,coords)\
    reduction(+:p)
  for (i=0; i < nrow; ++i) {
    d2 = 0;
    for (j=0; j < ncol; ++j) {
      d = coords[i][j] - xs[j];
      d2 += d*d;
    }
    if (d2 < dist2) {
      p += probs[i];
    }
  }
  return p;
}

std::list<Sample>
sample_n(unsigned int n
       , const std::vector<double>& probs
       , const std::vector<std::pair<float,float>>& min_max_coords
       , const std::vector<std::vector<float>>& coords
       , float radius) {
  // initialize random number generator
  auto dice = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                      , std::mt19937(std::random_device()()));
  // prepare sampling
  std::list<Sample> samples;
  float dist2 = radius*radius;
  unsigned int ncol = min_max_coords.size();
  // sampling
  for (unsigned int i=0; i < n; ++i) {
    // get coordinates
    std::vector<float> xs(ncol);
    for (unsigned int j=0; j < ncol; ++j) {
      xs[j] = dice()
            * (min_max_coords[j].second - min_max_coords[j].first)
            + min_max_coords[j].first;
    }
    // get probability
    double p = prob_estimate(xs
                           , dist2
                           , probs
                           , coords);
    // store sample
    samples.emplace(samples.begin(), p, xs);
  }
  // prob-descending ordering
  samples.sort([](Sample a, Sample b) -> bool {
    return a.first > b.first;
  });
  return samples;
}




StateSampler::StateSampler(const std::vector<unsigned int>& states
                         , const std::vector<unsigned int>& pops
                         , const std::vector<std::vector<float>>& ref_coords
                         , float radius)
  : _states(states)
  , _pops(pops)
  , _ref_coords(ref_coords)
  , _radius(radius) {
  // split probs / coords into states
  std::unordered_set<unsigned int> state_names(states.begin()
                                             , states.end());
  std::unordered_map<unsigned int
                   , std::vector<unsigned int>> pops_splitted;
  for (unsigned int s: state_names) {
    pops_splitted[s] = {};
    _ref_coords_splitted[s] = {};
  }
  for (unsigned int i=0; i < pops.size(); ++i){
    pops_splitted[states[i]].push_back(pops[i]);
    _ref_coords_splitted[states[i]].push_back(ref_coords[i]);
  }
  // - normalize pops -> probs;
  // - store minima/maxima of ref.-coords. for sampling
  // - initialize sampling pool
  for (unsigned int s: state_names) {
    _probs[s] = sum1_normalized(pops_splitted[s]);
    _min_max[s] = col_min_max(_ref_coords_splitted[s]);
    _sampling_pool[s] = _get_new_samples(s, pool_size_total);
    _sampling_pool_prob_sum[s] = sum_probs(_sampling_pool[s]);
  }
}

std::list<Sample>
StateSampler::_get_new_samples(unsigned int state
                             , unsigned int sample_size) {
  return sample_n(sample_size
                , _probs[state]
                , _min_max[state]
                , _ref_coords_splitted[state]
                , _radius);
}

Sample
StateSampler::operator()(unsigned int state) {
  // initialize random number generator
  auto dice = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                      , std::mt19937(std::random_device()()));
  // refill sampling pool if drained to minimum
  if (_sampling_pool[state].size() < pool_size_min) {
    _sampling_pool[state].merge(_get_new_samples(state
                                               , pool_size_min)
                              , [](Sample a, Sample b) {
                                  return a.first > b.first;
                                });
    _sampling_pool_prob_sum[state] = sum_probs(_sampling_pool[state]);
  }
  // sample from the pool (without replacement)
  double rnd = dice() * _sampling_pool_prob_sum[state];
  double running_probsum = 0;
  Sample sample;
  bool got_no_result = true;
  for (auto it=_sampling_pool[state].begin()
     ; it != _sampling_pool[state].end()
     ; ++it) {
    running_probsum += it->first;
    if (rnd < running_probsum) {
      sample = (*it);
      _sampling_pool[state].erase(it);
      got_no_result = false;
      break;
    }
  }
  if (got_no_result) {
    // this may happen, e.g. from numeric inaccuracies...
    sample = (*_sampling_pool[state].rbegin());
    _sampling_pool[state].pop_back();
  }
  _sampling_pool_prob_sum[state] -= sample.first;
  return sample;
}

