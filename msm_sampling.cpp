/*
Copyright (c) 2017, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>

#include <omp.h>
#include <boost/program_options.hpp>

#include "coords_file/coords_file.hpp"

#include "msm_sampling.hpp"
#include "tools.hpp"
#include "tools_io.hpp"
#include "probdist.hpp"




MetropolisStepper::MetropolisStepper() {
  // initialize random number generator
  _dice = std::bind(std::uniform_real_distribution<float>(0.0, 1.0)
                  , std::mt19937(std::random_device()()));
}

ScaledHypercubeStepper::ScaledHypercubeStepper(float step_width
                                             , std::vector<float> scaling)
  : MetropolisStepper()
  , _step_width(step_width)
  , _scaling(scaling) {
}

std::vector<float>
ScaledHypercubeStepper::operator()() {
  std::vector<float> step(_scaling.size());
  // randomized step drawn from [-step_width, +step_width]
  // per dimension with additional scaling.
  for (unsigned int i=0; i < _scaling.size(); ++i) {
    float w = _scaling[i]*_step_width;
    step[i] = _dice()*2*w - w;
  }
  return step;
}

HypercubeStepper::HypercubeStepper(float step_width
                                 , unsigned int n_dim)
  : ScaledHypercubeStepper(step_width
                         , std::vector<float>(n_dim, 1.0f)) {
}

ScaledHypersphereStepper::ScaledHypersphereStepper(float step_width
                                                 , std::vector<float> scaling)
  : MetropolisStepper()
  , _step_width(step_width)
  , _scaling(scaling) {
}

std::vector<float>
ScaledHypersphereStepper::operator()() {
  std::vector<float> step(_scaling.size());
  float p2 = 0.0f;
  // randomized step of +/- rnd([0,1])*step_width
  // on hypersphere (i.e. limited radius, randomly chosen)
  // with additional scaling per dimension.
  for (unsigned int i=0; i < _scaling.size(); ++i) {
    float w = _step_width;
    step[i] = _dice()*2*w - w;
    p2 += step[i] * step[i];
  }
  // normalize vector, scale radius by step width and random factor
  p2 = 1.0f/std::sqrt(p2) * _step_width * _dice();
  for (unsigned int i=0; i < _scaling.size(); ++i) {
    // additional scaling per dimension
    step[i] *= p2 * _scaling[i];
  }
  return step;
}

HypersphereStepper::HypersphereStepper(float step_width
                                     , unsigned int n_dim)
  : ScaledHypersphereStepper(step_width
                           , std::vector<float>(n_dim, 1.0f)) {
}

////



StateSampler::StateSampler(const std::vector<unsigned int>& states
                         , const std::vector<float>& fe
                         , const std::vector<std::vector<float>>& ref_coords
                         , float radius
                         , MetropolisStepper& stepper)
  : _states(states)
  , _fe(fe)
  , _ref_coords(ref_coords)
  , _radius(radius)
  , _radius_squared(radius*radius)
  , _stepper(stepper)
  , _n_frames_sampled(0) {
  // initialize random number generator
  _dice = std::bind(std::uniform_real_distribution<double>(0.0, 1.0)
                  , std::mt19937(std::random_device()()));
  // construct set of state names
  std::unordered_set<unsigned int> state_names(states.begin()
                                             , states.end());
  // split free energies and coodinates according to state
  for (unsigned int s: state_names) {
    _fe_splitted[s] = {};
    _ref_coords_splitted[s] = {};
  }
  for (unsigned int i=0; i < fe.size(); ++i){
    _fe_splitted[states[i]].push_back(fe[i]);
    _ref_coords_splitted[states[i]].push_back(ref_coords[i]);
  }
  // minima/maxima of ref.-coords. for sampling
  for (unsigned int s: state_names) {
    _min_max[s] = col_min_max(_ref_coords_splitted[s]);
    _max_fe_splitted[s] = (*std::max_element(_fe_splitted[s].begin()
                                           , _fe_splitted[s].end()));
  }
  _n_dim = ref_coords[0].size();
}


//TODO: idea: use simulation temp + ref. temp to scale according to temperature


Sample
StateSampler::operator()(unsigned int state) {
  std::vector<float> new_sample_coords(_n_dim);
  float new_sample_fe;
  auto new_fe = [&]() -> float {
    return fe_estimate(new_sample_coords
                     , _radius_squared
                     , _fe_splitted[state]
                     , _ref_coords_splitted[state]);
  };
  if (state == _prev_state) {
    bool no_sample_found = true;
    while (no_sample_found) {
      // next sampling step
      std::vector<float> step = _stepper();
      for (unsigned int i=0; i < _n_dim; ++i) {
        new_sample_coords[i] = _prev_sample.second[i] + step[i];
      }
      // Metropolis sampling:
      //   accept coords if smaller energy than previous
      //   or if Boltzmann weight of energy-difference is higher
      //   than random value.
      new_sample_fe = new_fe();
      if (new_sample_fe < _max_fe_splitted[state]) {
        if (new_sample_fe < _prev_sample.first
         || _dice() < std::min(1.0f
                             , std::exp(_prev_sample.first - new_sample_fe))) {
          no_sample_found = false;
        }
      }
    }
  } else {
    bool no_sample_found = true;
    while (no_sample_found) {
      // sample new coordinates 'out of the blue'
      for (unsigned int i=0; i < _n_dim; ++i) {
        new_sample_coords[i] = _min_max[state][i].first
                             + _dice()
                             * (_min_max[state][i].second
                              - _min_max[state][i].first);
      }
      // always accept new sample as long as energy < max(state_energy)
      new_sample_fe = new_fe();
      if (new_sample_fe < _max_fe_splitted[state]) {
        no_sample_found = false;
      }
    }
  }
  // bookkeeping for Metropolis algorithm
  ++_n_frames_sampled;
  Sample new_sample = {new_sample_fe
                     , new_sample_coords};
  _prev_sample = new_sample;
  _prev_state = state;
  return new_sample;
}








int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc(std::string(argv[1]).append(
    "\n\n"
    "sample from probability space of MSM new frames to\n"
    "generate a microscopic description of a protein's dynamics.\n"
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
     "show this help.")
    // required inputs
    ("traj,t", b_po::value<std::string>()->required(),
     "input (required): simulated state trajectory.")
    ("states,s", b_po::value<std::string>()->required(),
     "input (required): reference state trajectory.")
    ("free-energies,f", b_po::value<std::string>()->required(),
     "input (required): reference per-frame free energies.")
    ("coords,c", b_po::value<std::string>()->required(),
     "input (required): reference coordinates.")
    ("radius,r", b_po::value<float>()->required(),
     "input (required)  radius for probability integration.")
    // options
    ("output,o", b_po::value<std::string>()->default_value(""),
     "output:           the sampled coordinates / probability estimates."
     " default: stdout.")
    ("append-fe", b_po::bool_switch()->default_value(false),
     "                  append free energy estimate as additional"
     " column to the output coordinates")
    ("verbose,v", b_po::bool_switch()->default_value(false),
     "                  give verbose output.")
    ("nthreads,n", b_po::value<int>()->default_value(0),
     "                  number of OpenMP threads. default: 0; i.e. use"
     " OMP_NUM_THREADS env-variable.")
  ;
  // parse cmd arguments           
  try {
    b_po::store(b_po::command_line_parser(argc, argv)
                  .options(desc)
                  .run()
              , args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments:\n\n"
                << e.what()
                << "\n\n"
                << std::endl;
    }
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }

//TODO: secure against missing options

  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // setup OpenMP
  int n_threads = 0;
  if (args.count("nthreads")) {
    n_threads = args["nthreads"].as<int>();
  }
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // input (coordinates)
  std::string fname_out = args["output"].as<std::string>();
  float radius = args["radius"].as<float>();
  std::vector<std::vector<float>> ref_coords;
  unsigned int n_dim = 0;
  {
    CoordsFile::FilePointer fh =
      CoordsFile::open(args["coords"].as<std::string>()
                     , "r");
    while ( ! fh->eof()) {
      std::vector<float> buf = fh->next();
      if (buf.size() > 0) {
        ref_coords.push_back(buf);
      }
    }
    bool no_ref_coords = false;
    if (ref_coords.size() == 0) {
      no_ref_coords = true;
    } else {
      n_dim = ref_coords[0].size();
    }
    if (no_ref_coords
     || n_dim == 0) {
      std::cerr << "error: empty reference coordinates file" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // input (MSM trajectory to be sampled)
  std::vector<unsigned int> traj
    = read_states(args["traj"].as<std::string>());
  // prepare output file (or stdout)
  bool use_stdout;
  CoordsFile::FilePointer fh_out;
  if (fname_out == "") {
    use_stdout = true;
  } else {
    use_stdout = false;
    fh_out = CoordsFile::open(fname_out, "w");
  }
  //TODO test other stepper functions
  auto metropolis_stepper = HypersphereStepper(radius
                                             , n_dim);
  // state sampler function: state id -> sample
  StateSampler state_sampler(read_states(args["states"].as<std::string>())
                           , read_fe(args["free-energies"].as<std::string>())
                           , ref_coords
                           , radius
                           , metropolis_stepper);
  // helper function: sample -> stdout
  auto print_stdout = [&] (std::vector<float> xs) {
    for (float x: xs) {
      std::cout << " " << x;
    }
    std::cout << std::endl;
  };
  bool verbose = args["verbose"].as<bool>();
  bool append_fe = args["append-fe"].as<bool>();
  // sample the trajectory
  unsigned int i_state = 0;
  for (unsigned int state: traj) {
    if (verbose) {
      ++i_state;
      if (i_state % 100 == 0) {
        std::cerr << i_state << " / " << traj.size() << std::endl;
      }
    }
    // get new sample (pair of {free energy estimate, coordinates})
    Sample sample = state_sampler(state);
    if (append_fe) {
      // append free energy value to coordinates
      // as last column in output
      sample.second.push_back(sample.first);
    }
    if (use_stdout) {
      print_stdout(sample.second);
    } else {
      fh_out->write(sample.second);
    }
  }
  return EXIT_SUCCESS;
}

