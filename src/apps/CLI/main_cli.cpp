//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file main_cli.cpp
 * @brief Command-Line Interface version of the VQE Simulator.
 */

#include "core/ansatz.hpp"
#include "core/compat.h"
#include "core/logger.hpp"
#include "core/physics.hpp"
#include "core/simulation.hpp"
#include "core/adapt_sim.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Global flag for Ctrl-C termination
std::atomic<bool> keep_running{true};

void signal_handler(int signum) {
  spdlog::warn("Interruption (Ctrl-C) detected, stopping now...");
  keep_running.store(false);
}

// Helper to map string optimizer to nlopt::algorithm
nlopt::algorithm get_nlopt_algorithm(const std::string &opt_str) {
  if (opt_str == "LN_NELDERMEAD" || opt_str == "Nelder-Mead")
    return nlopt::LN_NELDERMEAD;
  if (opt_str == "LN_COBYLA" || opt_str == "COBYLA")
    return nlopt::LN_COBYLA;
  if (opt_str == "LN_BOBYQA" || opt_str == "BOBYQA")
    return nlopt::LN_BOBYQA;
  if (opt_str == "LN_NEWUOA")
    return nlopt::LN_NEWUOA;
  if (opt_str == "LN_NEWUOA_BOUND")
    return nlopt::LN_NEWUOA_BOUND;
  if (opt_str == "LN_PRAXIS")
    return nlopt::LN_PRAXIS;
  if (opt_str == "LN_SBPLX")
    return nlopt::LN_SBPLX;
  if (opt_str == "GN_DIRECT")
    return nlopt::GN_DIRECT;
  if (opt_str == "GN_DIRECT_L")
    return nlopt::GN_DIRECT_L;
  if (opt_str == "GN_CRS2_LM")
    return nlopt::GN_CRS2_LM;
  if (opt_str == "GN_ISRES")
    return nlopt::GN_ISRES;
  if (opt_str == "GN_ESCH")
    return nlopt::GN_ESCH;
  if (opt_str == "LD_LBFGS" || opt_str == "L-BFGS")
    return nlopt::LD_LBFGS;
  if (opt_str == "LD_SLSQP" || opt_str == "SLSQP")
    return nlopt::LD_SLSQP;
  return nlopt::LN_NELDERMEAD; // Default
}

// Helper to replace hyphens with underscores and convert to lowercase
std::string format_mapping(std::string mapping) {
  std::transform(mapping.begin(), mapping.end(), mapping.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::replace(mapping.begin(), mapping.end(), '-', '_');
  return mapping;
}

int main(int argc, char **argv) {
  // Register signal handler
  std::signal(SIGINT, signal_handler);

  // Initialize logger
  qb_log::init_logger();

  // Initialize QuEST
  initQuESTEnv();

  CLI::App app{"XRW-VQE Simulator CLI"};

  // Molecule & Hamiltonian options
  std::string opt_atom;
  std::string opt_basis = "sto-3g";
  int opt_charge = 0;
  int opt_spin = 0;
  std::string opt_mapping = "jordan_wigner";

  app.add_option("--atom", opt_atom,
                 "Molecule string (e.g. \"H 0 0 0; H 0 0 0.735\")")
      ->required();
  app.add_option("--basis", opt_basis, "Basis set");
  app.add_option("--charge", opt_charge, "Molecule charge");
  app.add_option("--spin", opt_spin, "Molecule spin (2S)");
  app.add_option("--mapping", opt_mapping, "Mapping type");

  // VQE options
  std::string opt_optimizer = "LN_NELDERMEAD";
  int opt_max_iter = 1000;
  double opt_tolerance = 1e-6;
  int opt_shots = 0;
  std::string opt_ansatz = "HEA";
  int opt_hea_depth = 1;

  std::string opt_grad_method = "psr";
  double opt_finite_tol = 1e-4;

  app.add_option("--optimizer", opt_optimizer, "NLopt algorithm or SPSA");
  app.add_option("--max-iter", opt_max_iter, "Maximum number of evaluations");
  app.add_option("--tolerance", opt_tolerance, "Relative tolerance");
  app.add_option("--shots", opt_shots,
                 "Number of shots for noise (0 = statevector)");
  app.add_option("--ansatz", opt_ansatz, "Ansatz type (HEA or UCCSD)");
  app.add_option("--hea-depth", opt_hea_depth, "Depth if Ansatz = HEA");
  app.add_option("--grad-method", opt_grad_method, "Gradient Method (fd, psr, gpsr)")->check(CLI::IsMember({"fd", "psr", "gpsr"}));
  app.add_option("--finite-tol", opt_finite_tol, "Tolerance for finite differences");

  // SPSA specific options
  double opt_spsa_a = 0.1;
  double opt_spsa_c = 0.1;
  double opt_spsa_A = 10.0;
  double opt_spsa_alpha = 0.602;
  double opt_spsa_gamma = 0.101;

  app.add_option("--spsa-a", opt_spsa_a, "SPSA step size numerator a");
  app.add_option("--spsa-c", opt_spsa_c, "SPSA perturbation numerator c");
  app.add_option("--spsa-A", opt_spsa_A, "SPSA stability constant A");
  app.add_option("--spsa-alpha", opt_spsa_alpha, "SPSA step size decay exponent alpha");
  app.add_option("--spsa-gamma", opt_spsa_gamma, "SPSA perturbation decay exponent gamma");

  // Diffraction Data options
  std::string opt_integrals = "";
  std::string opt_factors = "";
  double opt_lambda = 1.0;

  app.add_option("--integrals", opt_integrals, "Path to integrals file");
  app.add_option("--factors", opt_factors, "Path to experimental factors file");
  app.add_option("--lambda", opt_lambda, "Diffraction penalty lambda factor");

  // Output options
  std::string opt_out = "";
  app.add_option("--out", opt_out, "Explicit path for JSON output file");

  // ADAPT options
  bool opt_adapt = false;
  double opt_adapt_tol = 1e-3;
  int opt_adapt_max_iter = 10;
  app.add_flag("--adapt", opt_adapt, "Enable ADAPT-VQE");
  app.add_option("--adapt-tol", opt_adapt_tol, "Gradient L2 norm tolerance for ADAPT-VQE");
  app.add_option("--adapt-max-iter", opt_adapt_max_iter, "Maximum number of ADAPT-VQE macro iterations");

  // Warm start options
  std::string opt_warm_start = "";
  app.add_option("--warm-start", opt_warm_start,
                 "Path to warm start parameters JSON file");

  CLI11_PARSE(app, argc, argv);

  try {
    spdlog::info("XRW-VQE Simulator CLI starting...");

    std::string formatted_mapping = format_mapping(opt_mapping);

    // 1. Generate Hamiltonian via Python
    spdlog::info(">>> Generating Hamiltonian...");
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 python/generate_hamiltonian.py"; // assumes available in
                                                         // current dir
    command += " --atom \"" + opt_atom + "\"";
    command += " --basis " + opt_basis;
    command += " --charge " + std::to_string(opt_charge);
    command += " --spin " + std::to_string(opt_spin);
    command += " --mapping " + formatted_mapping;

    spdlog::info("CMD: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (!pipe) {
      spdlog::critical("Failed to open pipe for Hamiltonian generation.");
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
      std::string msg(buffer);
      while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
      }
      if (!msg.empty()) {
        spdlog::info("PySCF: {}", msg);
      }
    }

    int ret = _pclose(pipe);
    if (ret != 0) {
      spdlog::critical("Failed to execute Python script. Code: {}", ret);
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    // Parse hamiltonian.json
    std::ifstream f("hamiltonian.json");
    int num_paulis = 0;
    int num_qubits = 0;
    long long hilbert_space = 0;
    if (f.good()) {
      nlohmann::json hj;
      f >> hj;
      if (hj.contains("error")) {
        spdlog::critical("Python script error: {}",
                         hj["error"].get<std::string>());
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }
      for (auto &[key, val] : hj.items()) {
        if (key.find("term_") == 0)
          num_paulis++;
      }
      if (hj.contains("n_qubits")) {
        num_qubits = hj["n_qubits"].get<int>();
        hilbert_space = (long long)std::pow(2, num_qubits);
      }
    } else {
      spdlog::critical("Hamiltonian file hamiltonian.json not found.");
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    spdlog::info("Hamiltonian loaded. Qubits: {}, Terms: {}", num_qubits,
                 num_paulis);

    // 2. Load Physics
    Physics physics("hamiltonian.json");

    // 3. Setup Ansatz
    std::unique_ptr<Ansatz> ansatz;
    if (opt_ansatz == "HEA") {
      ansatz = std::make_unique<HEA>(physics.get_num_qubits(), opt_hea_depth);
    } else if (opt_ansatz == "UCCSD") {
      ansatz =
          std::make_unique<UCCSD>(physics.get_num_qubits(),
                                  physics.get_n_electrons(), formatted_mapping);
    } else {
      spdlog::critical("Ansatz type unknown: {}", opt_ansatz);
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    // 4. Create VQEContext
    // If not ADAPT, we setup Context and Simulation right here.
    // If ADAPT, we defer to ADAPT_sim.
    std::unique_ptr<VQEContext> ctx;
    std::unique_ptr<Simulation> sim;

    GradientMethod chosen_grad_method = GradientMethod::PSR;
    if (opt_grad_method == "fd") chosen_grad_method = GradientMethod::FD;
    else if (opt_grad_method == "gpsr") chosen_grad_method = GradientMethod::gPSR;

    if (!opt_adapt) {
      ctx = std::make_unique<VQEContext>(physics, *ansatz);
      ctx->n_shots = opt_shots;
      ctx->lambda = opt_lambda;
      
      ctx->grad_method = chosen_grad_method;
      ctx->fd_tol = opt_finite_tol;
      
      ctx->setup(opt_factors, opt_integrals);

      // 5. Create Simulation
      nlopt::algorithm algo = get_nlopt_algorithm(opt_optimizer);
      sim = std::make_unique<Simulation>(*ctx, algo);

      if (opt_optimizer == "SPSA") {
        sim->set_optimizer_type(Simulation::OptType::SPSA);
        SPSAParams params;
        params.a = opt_spsa_a;
        params.c = opt_spsa_c;
        params.A = opt_spsa_A;
        params.alpha = opt_spsa_alpha;
        params.gamma = opt_spsa_gamma;
        sim->set_spsa_params(params);
      } else {
        sim->set_optimizer_type(Simulation::OptType::NLOPT);
      }

      sim->set_max_evals(opt_max_iter);
      sim->set_tolerance(opt_tolerance);
    }

    // Required history arrays
    std::vector<double> iter_history;
    std::vector<double> energy_history;
    std::vector<double> base_energy_history;
    std::vector<double> chi_squared_history;
    std::vector<std::vector<double>> probs_history;
    std::vector<std::vector<double>> params_history;

    iter_history.reserve(opt_max_iter);
    energy_history.reserve(opt_max_iter);
    base_energy_history.reserve(opt_max_iter);
    chi_squared_history.reserve(opt_max_iter);
    probs_history.reserve(opt_max_iter);
    params_history.reserve(opt_max_iter);

    double best_energy = 1e9;
    std::vector<double> counts_values;

    // Callback
    auto callback = [&](int iter, double total_energy, double quantum_energy,
                        double chi_squared, const std::vector<double> &probs,
                        const std::vector<double> &cb_params) {
      if (!keep_running.load()) {
        throw std::runtime_error("Interrupted by user");
      }
      iter_history.push_back((double)iter);
      energy_history.push_back(total_energy);
      base_energy_history.push_back(quantum_energy);
      chi_squared_history.push_back(chi_squared);
      probs_history.push_back(probs);
      params_history.push_back(cb_params);

      counts_values = probs;
      if (total_energy < best_energy)
        best_energy = total_energy;

      if (iter % 10 == 0 ||
          iter == 1) { // Log every 10 iterations to not flood terminal
        spdlog::info("Iter: {}, Energy: {:.6f}, Base E: {:.6f}, Chi2: {:.6f}",
                     iter, total_energy, quantum_energy, chi_squared);
      }
    };

    spdlog::info(">>> Starting XRW-VQE... (Simulation)");
    std::vector<double> params(ansatz->get_num_params(), 0);

    // Small random perturbation to break zero-initialization symmetry
    // (avoids vanishing PSR gradients at the Hartree-Fock saddle point)
    {
      std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<double> dist(-1e-3, 1e-3);
      for (auto& p : params) p = dist(rng);
      spdlog::info("Initial parameters perturbed with uniform noise in [-0.001, 0.001] rad");
    }

    // --- Warm Start: Load parameters from file if provided ---
    if (!opt_warm_start.empty()) {
      std::ifstream ws_file(opt_warm_start);
      if (!ws_file.good()) {
        spdlog::critical("Warm start file not found: {}", opt_warm_start);
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }

      nlohmann::json ws_json;
      ws_file >> ws_json;

      // Validate ansatz tag
      auto tag = ws_json["ansatz_tag"];
      std::string ws_type = tag["type"].get<std::string>();
      int ws_nq = tag["num_qubits"].get<int>();

      if (ws_type != opt_ansatz) {
        spdlog::critical(
            "Warm start ansatz mismatch: file={}, current={}", ws_type,
            opt_ansatz);
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }
      if (ws_nq != num_qubits) {
        spdlog::critical(
            "Warm start num_qubits mismatch: file={}, current={}", ws_nq,
            num_qubits);
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }
      if (ws_type == "HEA" &&
          tag["depth"].get<int>() != opt_hea_depth) {
        spdlog::critical("Warm start HEA depth mismatch: file={}, current={}",
                         tag["depth"].get<int>(), opt_hea_depth);
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }
      if (ws_type == "UCCSD" &&
          tag["num_electrons"].get<int>() !=
              physics.get_n_electrons()) {
        spdlog::critical(
            "Warm start UCCSD num_electrons mismatch: file={}, current={}",
            tag["num_electrons"].get<int>(), physics.get_n_electrons());
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }

      // Validate parameter count
      auto ws_params = ws_json["parameters"].get<std::vector<double>>();
      if ((int)ws_params.size() != ansatz->get_num_params()) {
        spdlog::critical(
            "Warm start params size mismatch: file={}, expected={}",
            ws_params.size(), ansatz->get_num_params());
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }

      params = ws_params;
      spdlog::info("Warm start loaded: {} parameters from {}", params.size(),
                   opt_warm_start);
    }

    double noisy_energy = 0.0;
    std::string status_message = "XRW-VQE terminated.";

    try {
      if (opt_adapt) {
        ADAPT_sim adapt_sim(physics,
                            opt_optimizer, opt_max_iter, opt_tolerance, opt_adapt_tol,
                            opt_shots, opt_lambda, chosen_grad_method, opt_finite_tol,
                            opt_factors, opt_integrals,
                            opt_out, opt_adapt_max_iter,
                            callback);
        adapt_sim.run_adapt();
        status_message = "ADAPT-VQE terminated successfully.";
      } else {
        noisy_energy = sim->run(params, callback);

        // Re-fetch final probabilities using the sim method for output
        counts_values = sim->get_probabilities(params);

        // Log final optimized parameters
        std::string params_str = "[ ";
        for (size_t i = 0; i < params.size(); ++i) {
          params_str +=
              std::to_string(params[i]) + (i < params.size() - 1 ? ", " : " ]");
        }
        spdlog::info("Final optimized parameters: {}", params_str);
      }
    } catch (const std::exception &e) {
      spdlog::error("Simulation error: {}", e.what());
      status_message = "XRW-VQE terminated with error.";
    }

    if (opt_adapt) {
      finalizeQuESTEnv();
      return EXIT_SUCCESS;
    }

    // 5. Generate JSON strictly identical to GUI::SaveRun()
    nlohmann::json j;

    j["config"]["molecule"] = {
        {"atom_string", opt_atom},
        {"basis", opt_basis},
        {"charge", opt_charge},
        {"spin", opt_spin},
        {"mapping", opt_mapping} // GUI saves mappings[mapping_idx] which is e.g
                                 // "Jordan-Wigner", not the formatted one
    };

    j["config"]["vqe"] = {{"optimizer", opt_optimizer},
                          {"max_iterations", opt_max_iter},
                          {"shots", opt_shots},
                          {"hea_depth", opt_hea_depth},
                          {"ansatz", opt_ansatz}};

    j["results"] = {
        {"final_energy", noisy_energy},
        {"final_base_energy",
         base_energy_history.empty() ? 0.0 : base_energy_history.back()},
        {"final_chi_squared",
         chi_squared_history.empty() ? 0.0 : chi_squared_history.back()},
        {"best_exact_energy", best_energy},
        {"status", status_message}};

    // j["rdms"] = sim.get_rdms();

    std::vector<nlohmann::json> history_arr;
    for (size_t i = 0; i < iter_history.size(); ++i) {
      nlohmann::json entry;
      entry["iteration"] = iter_history[i];
      entry["energy"] = energy_history[i];
      if (i < base_energy_history.size())
        entry["base_energy"] = base_energy_history[i];
      if (i < chi_squared_history.size())
        entry["chi_squared"] = chi_squared_history[i];
      
      // On ne garde pas les probabilites par iteration pour ne pas surcharger le log
      // if (i < probs_history.size())
      //   entry["probabilities"] = probs_history[i];

      // On garde les parametres uniquement pour l'iteration finale
      if (i == iter_history.size() - 2) { //there's a mess with the very final iteration, take the previous one
        if (i < params_history.size())
          entry["parameters"] = params_history[i];
      }
      history_arr.push_back(entry);
    }
    j["history"] = history_arr;

    // j["state"]["probabilities"] = counts_values;
    // skip that part because of the high line count
    /*std::vector<std::string> labels;
    int n_q = 0;
    if (counts_values.size() > 0)
      n_q = (int)std::log2(counts_values.size());
    for (size_t i = 0; i < counts_values.size(); ++i) {
      std::string bitstring = "";
      for (int b = 0; b < n_q; ++b) {
        bitstring += ((i >> (n_q - 1 - b)) & 1) ? "1" : "0";
      }
      labels.push_back(bitstring);
    }
    j["state"]["labels"] = labels;*/

    j["system"] = {{"simulator", "XRW-VQE Simulator C++ v1.0"},
                   {"num_qubits", num_qubits},
                   {"num_paulis", num_paulis}};

    std::string filename = opt_out;
    if (filename.empty()) {
      std::time_t now = std::time(nullptr);
      char buf[100];
      std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
      filename = "run_" + std::string(buf) + ".json";
    }

    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;

    spdlog::info("Run saved to: {}", filename);

    // Save RDMs to a separate file to keep the main log clean
    std::string rdm_filename = filename;
    if (rdm_filename.find("run_") == 0) {
      rdm_filename.replace(0, 4, "rdms_");
    } else {
      rdm_filename = "rdms_" + filename;
    }
    std::ofstream rdm_out(rdm_filename);
    rdm_out << std::setw(4) << sim->get_rdms() << std::endl;
    spdlog::info("RDMs saved to: {}", rdm_filename);

    // --- Warm Start Export ---
    {
      nlohmann::json ws_json;
      nlohmann::json tag;
      tag["type"] = opt_ansatz;
      tag["num_qubits"] = num_qubits;
      if (opt_ansatz == "HEA") {
        tag["depth"] = opt_hea_depth;
      } else if (opt_ansatz == "UCCSD") {
        tag["num_electrons"] = physics.get_n_electrons();
      }
      ws_json["ansatz_tag"] = tag;
      ws_json["num_params"] = (int)params.size();
      ws_json["parameters"] = params;

      std::time_t ws_now = std::time(nullptr);
      char ws_buf[100];
      std::strftime(ws_buf, sizeof(ws_buf), "%Y%m%d_%H%M%S",
                    std::localtime(&ws_now));
      std::string ws_filename =
          "warm_start_" + std::string(ws_buf) + ".json";
      std::ofstream ws_out(ws_filename);
      ws_out << std::setw(4) << ws_json << std::endl;
      spdlog::info("Warm start saved to: {}", ws_filename);
    }

    finalizeQuESTEnv();

  } catch (const std::exception &e) {
    spdlog::critical("Critical error: {}", e.what());
    finalizeQuESTEnv();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
