
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

