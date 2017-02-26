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

#include <omp.h>
#include <boost/program_options.hpp>

#include "coords_file/coords_file.hpp"

#include "msm_sampling.hpp"
#include "tools.hpp"
#include "tools_io.hpp"
#include "probdist.hpp"

void
compute_fe_estimates(const std::vector<float>& ref_free_energies
                   , const std::vector<std::vector<float>>& ref_coords
                   , float radius
                   , std::string fname_out) {
  unsigned int nrow = ref_coords.size();
  float dist2 = radius*radius;
  Tools::IO::set_out(fname_out);

  namespace clk = std::chrono;
  clk::steady_clock::time_point start = clk::steady_clock::now();
  for (unsigned int i=0; i < nrow; ++i) {
    if (i % 100 == 0) {
      clk::steady_clock::time_point now = clk::steady_clock::now();
      Tools::IO::err() << clk::duration_cast<clk::seconds> (now-start).count()
                       << " secs:  "
                       << i << " / " << nrow
                       << std::endl;
    }
    float fe = fe_estimate(ref_coords[i]
                         , dist2
                         , ref_free_energies
                         , ref_coords);
    Tools::IO::out() << fe << std::endl;
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
    ("free-energies,f", b_po::value<std::string>()->required(),
     "input (required): reference per-frame free energies.")
    ("coords,c", b_po::value<std::string>()->required(),
     "input (required): reference coordinates.")
    ("output,o", b_po::value<std::string>()->default_value(""),
     "output:           the sampled coordinates / probability estimates."
     " default: stdout.")
    ("radius,r", b_po::value<float>()->default_value(0.1),
     "                  radius for probability integration. default: 0.1")
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
  // state sampler function: state id -> sample
  StateSampler state_sampler(read_states(args["states"].as<std::string>())
                           , read_fe(args["free-energies"].as<std::string>())
                           , ref_coords
                           , radius);
  // helper function: sample -> stdout
  auto print_stdout = [&] (std::vector<float> xs) {
    for (float x: xs) {
      std::cout << " " << x;
    }
    std::cout << std::endl;
  };
  // sample the trajectory
  unsigned int i_state = 0;
  for (unsigned int state: traj) {
    //TODO only if verbose
    ++i_state;
    if (i_state % 100 == 0) {
      std::cerr << i_state << " / " << traj.size() << std::endl;
    }
    ////
    Sample sample = state_sampler(state);
    // TODO make that optional:
    // append free energy value to coordinates
    sample.second.push_back(sample.first);
    if (use_stdout) {
      print_stdout(sample.second);
    } else {
      fh_out->write(sample.second);
    }
  }
  return EXIT_SUCCESS;
}

