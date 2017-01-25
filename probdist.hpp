
#include <vector>
#include <deque>
#include <utility>

std::vector<float>
sample(const std::vector<std::pair<float,float>>& min_max
     , const std::vector<double>& probs
     , const std::vector<std::vector<float>>& coords);

std::deque<std::vector<float>>
sample_n(unsigned int n
       , std::vector<std::pair<float,float>> min_max
       , const std::vector<double>& probs
       , const std::vector<std::vector<float>>& coords
       , const float dist2_perc);

float
compute_dist2_cutoff(std::vector<std::pair<float, float>> mm
                   , float dist2_ratio);

double
squared_dist_prob(const std::vector<float>& xs
                , float dist2
                , const std::vector<double>& probs
                , const std::vector<std::vector<float>>& coords);

