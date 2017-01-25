#pragma once

#include <string>
#include <vector>



//TODO put into tools_io

std::vector<unsigned int>
read_states(std::string fname);

std::vector<unsigned int>
read_pops(std::string fname);


//// misc

template <typename NUM>
std::vector<double>
sum1_normalized(const std::vector<NUM>& pops);

std::vector<std::pair<float, float>>
col_min_max(const std::vector<std::vector<float>>& coords);


//// template implementations

#include "tools.hxx"

