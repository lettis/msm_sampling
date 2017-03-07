#pragma once

#include <vector>
#include <list>
#include <functional>
#include <unordered_map>
#include <utility>

float
fe_estimate(const std::vector<float>& xs
          , float dist2
          , const std::vector<float>& fe
          , const std::vector<std::vector<float>>& coords);

