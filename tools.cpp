
#include <iostream>
#include <fstream>
#include <limits>
#include <utility>

#include "tools.hpp"

namespace {

std::vector<unsigned int>
_read_col(std::string fname) {
  std::vector<unsigned int> col;
  std::ifstream ifs(fname);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << fname << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (ifs.good()) {
      unsigned int buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        col.push_back(buf);
      }
    }
  }
  return col;
}

} // end local namespace

std::vector<unsigned int>
read_states(std::string fname) {
  return _read_col(fname);
}

std::vector<unsigned int>
read_pops(std::string fname) {
  return _read_col(fname);
}


std::vector<std::pair<float, float>>
col_min_max(const std::vector<std::vector<float>>& coords) {
  unsigned int nrow = coords.size();
  if (nrow == 0) {
    std::cerr << "error: no sampling. cannot compute min/max of coordinates." << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned int ncol = coords[0].size();
  std::vector<std::pair<float, float>> mm(ncol
                                        , {std::numeric_limits<float>::infinity()
                                         , -std::numeric_limits<float>::infinity()});
  for (unsigned int i=0; i < nrow; ++i) {
    for (unsigned int j=0; j < ncol; ++j) {
      float ref_val = coords[i][j];
      if (ref_val < mm[j].first) {
        mm[j].first = ref_val;
      }
      if (ref_val > mm[j].second) {
        mm[j].second = ref_val;
      }
    }
  }
  return mm;
}



