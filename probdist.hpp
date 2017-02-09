
#include <vector>
#include <list>
#include <functional>
#include <unordered_map>
#include <utility>

typedef std::pair<double, std::vector<float>> Sample;

double
sum_probs(const std::list<Sample>& samples);

std::list<Sample>
sample_n(unsigned int n
       , const std::vector<double>& probs
       , const std::vector<std::pair<float,float>>& min_max_coords
       , const std::vector<std::vector<float>>& coords
       , float radius);

double
prob_estimate(const std::vector<float>& xs
         , float dist2
         , const std::vector<double>& probs
         , const std::vector<std::vector<float>>& coords);


class StateSampler {
 public:
  StateSampler(const std::vector<unsigned int>& states
             , const std::vector<unsigned int>& pops
             , const std::vector<std::vector<float>>& ref_coords
             , float radius);
  Sample operator()(unsigned int state);

  static constexpr unsigned int pool_size_total = std::pow(2,11);
  static constexpr unsigned int pool_size_min = std::pow(2,10);

 private:
  const std::vector<unsigned int>& _states;
  const std::vector<unsigned int>& _pops;
  const std::vector<std::vector<float>>& _ref_coords;
  float _radius;

  std::unordered_map<unsigned int, std::vector<double>>
    _probs;
  std::unordered_map<unsigned int, std::vector<std::pair<float,float>>>
    _min_max;
  std::unordered_map<unsigned int, std::list<Sample>>
    _sampling_pool;
  std::unordered_map<unsigned int, double>
    _sampling_pool_prob_sum;
  std::unordered_map<unsigned int, std::vector<std::vector<float>>>
    _ref_coords_splitted;

  std::list<Sample> _get_new_samples(unsigned int state
                                   , unsigned int sample_size);
};

