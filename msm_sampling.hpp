#pragma once

#include <vector>
#include <string>
#include <unordered_map>


typedef std::pair<float, std::vector<float>> Sample;



//// various stepping functions (as functors) for Metropolis sampling.

class MetropolisStepper {
 public:
  MetropolisStepper();
  virtual std::vector<float> operator()() = 0;
 protected:
  std::function<float()> _dice;
};

class ScaledHypercubeStepper : public MetropolisStepper {
 public:
  ScaledHypercubeStepper(float step_width
                       , std::vector<float> scaling);
  std::vector<float> operator()();
 protected:
  float _step_width;
  std::vector<float> _scaling;
};

class HypercubeStepper : public ScaledHypercubeStepper {
 public:
  HypercubeStepper(float step_width
                 , unsigned int n_dim);
};

class ScaledHypersphereStepper : public MetropolisStepper {
 public:
  ScaledHypersphereStepper(float step_width
                         , std::vector<float> scaling);
  std::vector<float> operator()();
 protected:
  float _step_width;
  std::vector<float> _scaling;
};

class HypersphereStepper : public ScaledHypersphereStepper {
 public:
  HypersphereStepper(float step_width
                   , unsigned int n_dim);
};

////////




class StateSampler {
 public:
  StateSampler(const std::vector<unsigned int>& states
             , const std::vector<float>& fe
             , const std::vector<std::vector<float>>& ref_coords
             , float radius
             , MetropolisStepper& stepper);
  Sample operator()(unsigned int state);

 private:
  // inits
  const std::vector<unsigned int>& _states;
  const std::vector<float>& _fe;
  const std::vector<std::vector<float>>& _ref_coords;
  float _radius;
  float _radius_squared;
  MetropolisStepper& _stepper;
  // bookkeeping
  unsigned int _n_frames_sampled;
  unsigned int _prev_state;
  Sample _prev_sample;
  // random number generator
  std::function<float()> _dice;
  // splitted reference values
  std::unordered_map<unsigned int, std::vector<float>> _fe_splitted;
  std::unordered_map<unsigned int, float> _max_fe_splitted;
  std::unordered_map<unsigned int, std::vector<std::pair<float,float>>>
    _min_max;
  std::unordered_map<unsigned int, std::vector<std::vector<float>>>
    _ref_coords_splitted;
  // dimensionality
  unsigned int _n_dim;
};


//void
//compute_fe_estimates(const std::vector<float>& ref_free_energies
//                   , const std::vector<std::vector<float>>& ref_coords
//                   , float radius
//                   , std::string fname_out);

//void
//sample_traj(const std::vector<unsigned int>& traj
//          , const std::vector<unsigned int>& states
//          , const std::vector<unsigned int>& pops
//          , const std::vector<std::vector<float>>& coords
//          , unsigned int batchsize
//          , float radius
//          , std::string fname_out);


