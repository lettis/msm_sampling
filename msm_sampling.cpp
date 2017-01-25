/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <vector>
#include <algorithm>
#include <chrono>

#include <omp.h>
#include <boost/program_options.hpp>

#include "coords_file/coords_file.hpp"

#include "msm_sampling.hpp"
#include "tools.hpp"
#include "tools_io.hpp"
#include "probdist.hpp"

void
compute_prob_estimates(const std::vector<unsigned int>& pops
                     , const std::vector<std::vector<float>>& ref_coords
                     , float dist2_ratio
                     , std::string fname_out) {
  unsigned int nrow = ref_coords.size();
  float dist2 = compute_dist2_cutoff(col_min_max(ref_coords)
                                   , dist2_ratio);
  std::vector<double> ref_probs = sum1_normalized(pops);
  Tools::IO::set_out(fname_out);

  namespace clk = std::chrono;
  clk::steady_clock::time_point start = clk::steady_clock::now();
  for (unsigned int i=0; i < nrow; ++i) {
    if (i % 100 == 0) {
      clk::steady_clock::time_point now = clk::steady_clock::now();
      Tools::IO::err() << clk::duration_cast<clk::seconds> (now-start).count()
                       << " secs:  "
                       << i << " / " << nrow << std::endl;
    }
    double prob_estimate = squared_dist_prob(ref_coords[i]
                                           , dist2
                                           , ref_probs
                                           , ref_coords);
    Tools::IO::out() << prob_estimate << std::endl;
  }
}


void
sample_traj(const std::vector<unsigned int>& traj
          , const std::vector<unsigned int>& states
          , const std::vector<unsigned int>& pops
          , const std::vector<std::vector<float>>& ref_coords
          , unsigned int batchsize
          , float dist2_ratio
          , std::string fname_out) {
  // split probs / coords into states
  std::unordered_set<unsigned int> state_names(states.begin()
                                             , states.end());
  std::unordered_map<unsigned int
                   , std::vector<unsigned int>> pops_splitted;
  std::unordered_map<unsigned int
                   , std::vector<std::vector<float>>> ref_coords_splitted;
  for (unsigned int s: state_names) {
    pops_splitted[s] = {};
    ref_coords_splitted[s] = {};
  }
  for (unsigned int i=0; i < pops.size(); ++i){
    pops_splitted[states[i]].push_back(pops[i]);
    ref_coords_splitted[states[i]].push_back(ref_coords[i]);
  }
  // normalize pops -> probs;
  // store minima/maxima of ref.-coords. for sampling
  std::unordered_map<unsigned int
                   , std::vector<double>> probs;
  std::unordered_map<unsigned int
                   , std::vector<std::pair<float,float>>> min_max;
  for (unsigned int s: state_names) {
    probs[s] = sum1_normalized(pops_splitted[s]);
    min_max[s] = col_min_max(ref_coords_splitted[s]);
  }
  // prepare output file (or stdout)
  bool use_stdout;
  CoordsFile::FilePointer fh_out;
  if (fname_out == "") {
    use_stdout = true;
  } else {
    use_stdout = false;
    fh_out = CoordsFile::open(fname_out, "w");
  }
  // sampling and output
  auto print_stdout = [&] (std::vector<float> xs) {
    for (float x: xs) {
      std::cout << " " << x;
    }
    std::cout << std::endl;
  };
  std::map<unsigned int
         , std::deque<std::vector<float>>> sampling_pool;
  for (unsigned int s: traj) {
    if (sampling_pool.count(s) == 0
     || sampling_pool[s].size() == 0) {
      std::cerr << "batch sampling of state " << s << " ..." << std::endl;
      sampling_pool[s] = sample_n(batchsize
                                , min_max[s]
                                , probs[s]
                                , ref_coords_splitted[s]
                                , dist2_ratio);
      std::cerr << "... done. computed " << batchsize << " samples." << std::endl;
    }
    if (use_stdout) {
      print_stdout(sampling_pool[s].front());
    } else {
      fh_out->write(sampling_pool[s].front());
    }
    sampling_pool[s].pop_front();
  }
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
    ("traj,t", b_po::value<std::string>(),
     "input:            simulated state trajectory.")
    ("states,s", b_po::value<std::string>(),
     "input:            reference state trajectory.")
    ("pops,p", b_po::value<std::string>()->required(),
     "input (required): reference per-frame populations.")
    ("coords,c", b_po::value<std::string>()->required(),
     "input (required): reference coordinates.")
    ("output,o", b_po::value<std::string>()->default_value(""),
     "output:           the sampled coordinates / probability estimates. default: stdout.")
    ("batchsize,b", b_po::value<unsigned int>()->default_value(1000),
     "batchsize of sample construction. default: 1000")
    ("ratio", b_po::value<float>()->default_value(0.01),
     "                  ratio of probability integrator size to total size. default: 0.01")
    ("estimates,e", b_po::bool_switch()->default_value(false),
     "flag:             compute estimates of local probabilities for reference coordinates. default: not set.")
    ("nthreads,n", b_po::value<int>()->default_value(0),
     "                  number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
  ;
  // parse cmd arguments           
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments:\n\n" << e.what() << "\n\n" << std::endl;
    }
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
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
  // input
  std::string fname_out = args["output"].as<std::string>();
  float dist2_ratio = args["ratio"].as<float>();
  std::vector<unsigned int> pops = read_pops(args["pops"].as<std::string>());
  std::vector<std::vector<float>> ref_coords;
  {
    CoordsFile::FilePointer fh = CoordsFile::open(args["coords"].as<std::string>()
                                                , "r");
    while ( ! fh->eof()) {
      std::vector<float> buf = fh->next();
      if (buf.size() > 0) {
        ref_coords.push_back(buf);
      }
    }
  }
  bool compute_estimates = args["estimates"].as<bool>();
  if (compute_estimates) {
    // compute probability estimates for reference coordinates
    compute_prob_estimates(pops
                         , ref_coords
                         , dist2_ratio
                         , fname_out);
  } else {
    // sample coordinates for state-trajectory
    // TODO: check for keys in args
    unsigned int batchsize = args["batchsize"].as<unsigned int>();
    std::vector<unsigned int> states = read_states(args["states"].as<std::string>());
    std::vector<unsigned int> traj = read_states(args["traj"].as<std::string>());
    sample_traj(traj
              , states
              , pops
              , ref_coords
              , batchsize
              , dist2_ratio
              , fname_out);
  }
  return EXIT_SUCCESS;
}

