
#include <vector>
#include <utility>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <map>

#include <omp.h>

#include "probdist.hpp"


float
compute_dist2_cutoff(std::vector<std::pair<float, float>> mm
                   , float dist2_ratio) {
  unsigned int ncol = mm.size();
  float dist2=0;
  for (unsigned int j=0; j < ncol; ++j) {
    float d = mm[j].second - mm[j].first;
    dist2 += d*d;
  }
  return dist2 * dist2_ratio;
}


double
squared_dist_prob(const std::vector<float>& xs
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



std::deque<std::vector<float>>
sample_n(unsigned int n
       , std::vector<std::pair<float,float>> min_max
       , const std::vector<double>& probs
       , const std::vector<std::vector<float>>& coords
       , const float dist2_ratio) {
  // initialize random number generator
  std::random_device rd;
  auto dice = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                      , std::mt19937(rd()));
  // prepare sampling
  std::deque<std::vector<float>> samples;
  float dist2 = compute_dist2_cutoff(min_max
                                   , dist2_ratio);
  unsigned int ncol = min_max.size();
  // sampling
  for (unsigned int i=0; i < n; ++i) {

    std::cerr << i << " / " << n << std::endl;

    std::vector<float> xs(ncol);
    bool sample_not_found = true;
    while (sample_not_found) {
      for (unsigned int j=0; j < ncol; ++j) {
        xs[j] = dice() * (min_max[j].second - min_max[j].first)
                  + min_max[j].first;
      }
      double p = squared_dist_prob(xs
                                 , dist2
                                 , probs
                                 , coords);
      if (dice() < p) {
        sample_not_found = false;
        samples.push_back(xs);
      }
    }
  }
  return samples;
}

