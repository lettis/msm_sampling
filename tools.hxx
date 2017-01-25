
#include "tools.hpp"

template <typename NUM>
std::vector<double>
sum1_normalized(const std::vector<NUM>& pops) {
  unsigned int nrow = pops.size();
  std::vector<double> probs(nrow);
  // compute sum of pops
  double sum_pops = 0;
  for (unsigned int i=0; i < nrow; ++i) {
    sum_pops += pops[i];
  }
  // sum-to-one normalized probabilities
  for (unsigned int i=0; i < nrow; ++i) {
    probs[i] = ((double) pops[i]) / sum_pops;
  }
  return probs;
}



