//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file simulation.cpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Implementation of the Simulation class for VQE optimization.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#define _USE_MATH_DEFINES
#include "simulation.hpp"
#include "compat.h"
#include <QuEST.h>
#include <bitset>
#include <cmath>
#include <complex>
#include <fstream> // Added by user instruction
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp> // Added by user instruction
#include <omp.h>
#include <random>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

//------------------------------------------------------------------------------
//     XORSHIFT64 (Fast PRNG)
//------------------------------------------------------------------------------

struct fast_xorshift64 {
  using result_type = uint64_t;
  uint64_t state;

  fast_xorshift64(uint64_t seed)
      : state(seed == 0 ? 0x1337C0DECAFEF00DULL : seed) {}

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  inline result_type operator()() {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state = x;
  }
};

//------------------------------------------------------------------------------
//     CONSTRUCTOR / DESTRUCTOR
//------------------------------------------------------------------------------

/**
 * @brief Constructs the Simulation object.
 *
 * Initializes the simulation environment, creates the quantum register (Qureg),
 * and configures the optimizer with default bounds and tolerance.
 *
 * @param physics Reference to the Physics object containing system properties.
 * @param ansatz Reference to the Ansatz object defining the variational
 * circuit.
 * @param algo The NLopt optimization algorithm to use.
 */
Simulation::Simulation(Physics &physics, Ansatz &ansatz, nlopt::algorithm algo)
    : physics(physics), ansatz(ansatz),
      optimizer(algo, ansatz.get_num_params()) {

  // Initialize the Quantum Entroypy format (QuEST) quantum register
  qubits = createQureg(physics.get_num_qubits());

  // Configure the non-linear optimizer parameters
  optimizer.set_ftol_rel(1e-8);
}

/**
 * @brief Destroys the Simulation object.
 *
 * Releases resources associated with the quantum register.
 */
Simulation::~Simulation() { destroyQureg(qubits); }

//------------------------------------------------------------------------------
//     COST FUNCTION
//------------------------------------------------------------------------------

/**
 * @brief The cost function to be minimized by the VQE algorithm.
 *
 * Calculates the expectation value of the Hamiltonian for the given parameters.
 * Supports both exact simulation and shot-noise simulation (if n_shots > 0).
 * Includes a penalty term to enforce particle number conservation.
 *
 * @param params The current variational parameters.
 * @param grad The gradient vector (unused in gradient-free algorithms).
 * @param data_ptr Pointer to VQEData structure containing simulation context.
 * @return double The estimated energy (cost) for the given parameters.
 */
double Simulation::evaluate_functional(const std::vector<double> &params,
                                       VQEData *data, Qureg local_qubits,
                                       std::vector<qcomp> &rdm1_out) {

  // Prepare rdm1_out vector for 1-RDM results
  rdm1_out.assign(data->rdm1_operators.size(), 0.0);

  // Reset the quantum register to the zero state |00...0>
  initZeroState(local_qubits);

  // Prepare the Hartree-Fock initial state by applying X gates to electron
  // positions
  for (int i = 0; i < data->n_electrons; ++i) {
    applyPauliX(local_qubits, i);
  }

  // Apply the parameterized variational circuit (Ansatz)
  data->ansatz.construct_circuit(local_qubits, params, data->paulis);

  //--------------------------------------------------------------------------
  // Particle Number Penalty Calculation
  //--------------------------------------------------------------------------
  qreal number_exp =
      data->n_electrons; // Default to expected number if preserved
  if (!data->ansatz.preserves_particle_number() && data->has_number_op) {
    number_exp = calcExpecPauliStrSum(local_qubits, data->number_op);
  }

  double energy = 0.0;
  double variance = 0.0;

  //--------------------------------------------------------------------------
  // Energy Evaluation
  //--------------------------------------------------------------------------

  if (data->n_shots > 0) {
    // --- Noisy Simulation (Shot Noise) ---
    // Simulates the effect of finite sampling shots by iterating over
    // Hamiltonian terms.

    const auto &coefficients = data->physics.get_coefficients();
    thread_local fast_xorshift64 gen(std::random_device{}());

    for (size_t i = 0; i < data->parsed_paulis.size(); ++i) {
      // Skip negligible coefficients to improve performance
      if (std::abs(coefficients[i]) < 1e-9)
        continue;

      // Use the pre-calculated PauliStrSum
      double expectation =
          calcExpecPauliStrSum(local_qubits, data->single_term_sums[i]);

      // Simulate sampling from a Binomial distribution based on exact
      // expectation P(+1) = (1 + <P>) / 2
      double p_plus = (1.0 + expectation) / 2.0;
      p_plus = std::max(0.0, std::min(1.0, p_plus));

      std::binomial_distribution<> binom(data->n_shots, p_plus);
      int n_plus = binom(gen);
      int n_minus = data->n_shots - n_plus;

      double expectation_est = (double)(n_plus - n_minus) / data->n_shots;

      // Accumulate energy contribution
      double c_real = coefficients[i].real();
      energy += c_real * expectation_est;

      // Accumulate variance contribution: Var(c * P_est) = c^2 * Var(P_est)
      // Var(P_est) = (1 - <P>^2) / N_shots
      variance +=
          c_real * c_real * (1.0 - expectation * expectation) / data->n_shots;
    }

    // Store statistical moments
    if (data->variance_ptr)
      *data->variance_ptr = variance;
    if (data->std_ptr)
      *data->std_ptr = std::sqrt(variance);

  } else {
    // --- Exact Simulation ---
    // Computes the exact expectation value using state vector operations
    energy = calcExpecPauliStrSum(local_qubits, data->hamiltonian);

    if (data->variance_ptr)
      *data->variance_ptr = 0.0;
    if (data->std_ptr)
      *data->std_ptr = 0.0;
  }

  // Apply quadratic penalty for deviations from expected electron number
  qreal penalty = 3.0 * std::pow((number_exp - data->n_electrons), 2);
  energy += penalty;

  //--------------------------------------------------------------------------
  // 1-RDM Evaluation
  //--------------------------------------------------------------------------
  for (size_t idx = 0; idx < data->rdm1_operators.size(); ++idx) {
    const auto &rdm_term = data->rdm1_operators[idx];
    qcomp term_expectation = 0.0;

    for (size_t s = 0; s < rdm_term.strings.size(); ++s) {
      if (std::abs(rdm_term.coeffs[s].real()) < 1e-9 &&
          std::abs(rdm_term.coeffs[s].imag()) < 1e-9) {
        continue;
      }

      double p_expect_real;
      if (data->n_shots > 0) {
        // In noisy sim we should ideally sample the 1-RDM individually as well,
        // we reuse exact for fast prototyping. Usually, 1-RDM is measured
        // during the same shot batch unless mapping requires separate bases.
      }

      qcomp one = 1.0;
      PauliStr p_arr[] = {rdm_term.strings[s]};
      PauliStrSum temp_sum = createPauliStrSum(p_arr, &one, 1);
      p_expect_real = calcExpecPauliStrSum(local_qubits, temp_sum);
      destroyPauliStrSum(temp_sum);

      term_expectation += rdm_term.coeffs[s] * p_expect_real;
    }
    rdm1_out[idx] = term_expectation;
  }

  if (!data->current_1rdm.empty() && data->integrals.size() > 0) {
    // Map the 1-RDM evaluated vector to an Eigen Vector
    Eigen::Map<Eigen::VectorXcd> rdm1_map(
        reinterpret_cast<std::complex<double> *>(data->current_1rdm.data()),
        data->current_1rdm.size());

    // Perform the matrix-vector multiplication to get the theoretical factors
    Eigen::VectorXcd calc_factors = data->integrals * rdm1_map;

    // Computation of eta
    double eta = ((calc_factors.cwiseAbs() * data->exp_factors.cwiseAbs())
                      .cwiseQuotient(data->uncertainties.cwiseAbs2())
                      .sum()) /
                 (calc_factors.cwiseAbs2()
                      .cwiseQuotient(data->uncertainties.cwiseAbs2())
                      .sum());

    // TODO: Here you will define how these calculated factors impact the total
    double chi_squared =
        (1.0 / data->exp_factors.size()) *
        (eta * calc_factors.cwiseAbs() - data->exp_factors.cwiseAbs())
            .cwiseAbs2()
            .cwiseQuotient(data->uncertainties.cwiseAbs2())
            .sum();
    energy += data->lambda * chi_squared;
  }

  return energy;
}

double Simulation::cost_function(const std::vector<double> &params,
                                 std::vector<double> &grad, void *data_ptr) {

  VQEData *data = static_cast<VQEData *>(data_ptr);

  // 1. Calculate the energy of the current point on the main register
  double base_energy =
      evaluate_functional(params, data, data->qubits, data->current_1rdm);

  // 2. Calculate the local gradients using Parameter Shift Rule (PSR)
  // ONLY if requested by the optimizer
  if (!grad.empty()) {
    int num_params = params.size();

    // Initialize Qureg
    Qureg local_qubits(num_params);

    // Evaluate parameter shifts sequentially to allow QuEST/Eigen to fully
    // utilize threads
    std::exception_ptr global_exception = nullptr;
    for (int i = 0; i < num_params; ++i) {
      if (global_exception)
        continue;

      try {
        std::vector<double> shifted_params = params;
        std::vector<qcomp> rdm1_plus;
        std::vector<qcomp> rdm1_minus;

        // Shift +π/2
        shifted_params[i] = params[i] + M_PI / 2.0;
        double e_plus =
            evaluate_functional(shifted_params, data, local_qubits, rdm1_plus);

        // Shift -π/2
        shifted_params[i] = params[i] - M_PI / 2.0;
        double e_minus =
            evaluate_functional(shifted_params, data, local_qubits, rdm1_minus);

        // Parameter Shift Rule formula
        grad[i] = 0.5 * (e_plus - e_minus);

        // Reset the Qureg
        initZeroState(local_qubits);
      } catch (...) {
        if (!global_exception) {
          global_exception = std::current_exception();
        }
      }
    }

    if (global_exception) {
      // Destroy Qureg before rethrowing
      destroyQureg(local_qubits);
      std::rethrow_exception(global_exception);
    }
    // Destroy Qureg
    destroyQureg(local_qubits);
  }

  // Execute callback if provided (e.g., for GUI updates)
  if (data->callback) {
    data->current_iter++;

    int num_qubits = data->num_qubits;
    long long dim = 1LL << num_qubits;
    std::vector<double> probs(dim);

    // Compute exact state vector probabilities
    for (long long j = 0; j < dim; ++j) {
      qcomp amp = getQuregAmp(data->qubits, j);
      probs[j] = amp.real() * amp.real() + amp.imag() * amp.imag();
    }

    // If noisy simulation, sample probabilities to reflect shot noise
    if (data->n_shots > 0) {
      std::vector<double> sampled_probs(dim, 0.0);
      std::mt19937 gen(std::random_device{}());
      std::discrete_distribution<> d(probs.begin(), probs.end());

      for (int k = 0; k < data->n_shots; ++k) {
        int outcome = d(gen);
        sampled_probs[outcome] += 1.0;
      }
      // Normalize counts to probabilities
      for (auto &p : sampled_probs) {
        p /= data->n_shots;
      }
      probs = sampled_probs;
    }

    data->callback(data->current_iter, base_energy, probs, params);
  }

  return base_energy;
}

//------------------------------------------------------------------------------
//     EXECUTION
//------------------------------------------------------------------------------

/**
 * @brief Executes the VQE optimization process.
 *
 * Configures the VQE data structure, sets the cost function, and runs the NLopt
 * optimizer.
 *
 * @param optimal_params Output vector to store the optimized parameters.
 * @param callback Function to be called at each iteration (step, energy,
 * probabilities).
 * @return double The minimum energy found after optimization.
 */
double
Simulation::run(std::vector<double> &optimal_params,
                std::function<void(int, double, const std::vector<double> &,
                                   const std::vector<double> &)>
                    callback,
                const std::string &fcalc_path, const std::string &ft_int_path) {

  // Retrieve the Hamiltonian in QuEST format
  PauliStrSum ham = physics.get_quest_hamiltonian();

  // Initialize VQE data context for the cost function
  spdlog::info("[Simulation] Starting VQE Optimization with {} params and max "
               "{} evaluations",
               ansatz.get_num_params(), optimizer.get_maxeval());

  VQEData data{ansatz,
               qubits,
               ham,
               physics.get_pauli_strings(),
               physics,
               callback,
               0,
               physics.get_n_electrons(),
               physics.get_num_qubits(),
               n_shots,
               lambda_val,
               &last_variance,
               &last_std};

  // Reset statistical trackers
  last_variance = 0.0;
  last_std = 0.0;

  // Pre-parse Pauli strings for noisy simulation to avoid parsing overhead in
  // the cost function
  std::vector<PauliStr> parsed_paulis;
  const auto &pauli_strings = physics.get_pauli_strings();
  parsed_paulis.reserve(pauli_strings.size());

  for (const auto &s : pauli_strings) {
    PauliStr pStr;
    if (s == "I") {
      std::vector<int> idx = {0};
      pStr = getPauliStr("I", idx);
    } else {
      std::string codes;
      std::vector<int> idx;
      std::stringstream ss(s);
      std::string token;
      while (ss >> token) {
        if (token.length() < 2)
          continue;
        codes += token[0];
        idx.push_back(std::stoi(token.substr(1)));
      }
      if (codes.empty()) {
        std::vector<int> z = {0};
        pStr = getPauliStr("I", z);
      } else {
        pStr = getPauliStr(codes, idx);
      }
    }
    parsed_paulis.push_back(pStr);
  }
  data.parsed_paulis = parsed_paulis;

  // Pre-calculate individual PauliStr sums for noisy simulations
  std::vector<PauliStrSum> single_sums;
  single_sums.reserve(parsed_paulis.size());
  for (const auto &pStr : parsed_paulis) {
    qcomp one = 1.0;
    PauliStr terms_arr[] = {pStr};
    single_sums.push_back(createPauliStrSum(terms_arr, &one, 1));
  }
  data.single_term_sums = single_sums;

  //----------------------------------------------------------------------------
  // 1-RDM Generation & Extraction
  //----------------------------------------------------------------------------
  std::map<std::pair<int, int>, int> rdm1_index_map;
  try {
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command +=
        "python3 python/generate_1rdm.py --n_qubits " +
        std::to_string(physics.get_num_qubits()) +
        " --mapping jordan_wigner"; // assuming JW for now, could be dynamic

    spdlog::info("[Simulation] Generating 1-RDM mapping: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (!pipe) {
      throw std::runtime_error("_popen() failed!");
    }

    char buffer[256];
    std::string json_filepath = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      json_filepath += buffer;
    }
    _pclose(pipe);

    // Remove trailing newlines explicitly
    json_filepath.erase(
        std::remove(json_filepath.begin(), json_filepath.end(), '\n'),
        json_filepath.end());
    json_filepath.erase(
        std::remove(json_filepath.begin(), json_filepath.end(), '\r'),
        json_filepath.end());

    std::ifstream f(json_filepath);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open 1-RDM JSON file: " +
                               json_filepath);
    }

    nlohmann::json rdm_json;
    f >> rdm_json;

    // Group identically (p,q) items
    std::map<std::pair<int, int>, RDM1Term> rdm_map;

    for (const auto &item : rdm_json) {
      int p = item["p"];
      int q = item["q"];
      double c_real = item["coeff_real"];
      double c_imag = item["coeff_imag"];
      std::string pauli_string = item["string"];

      std::pair<int, int> pq_pair = {p, q};
      if (rdm_map.find(pq_pair) == rdm_map.end()) {
        rdm_map[pq_pair] = {p, q, {}, {}};
      }

      rdm_map[pq_pair].strings.push_back(getPauliStr(pauli_string));
      rdm_map[pq_pair].coeffs.push_back({c_real, c_imag});
    }

    int current_idx = 0;
    for (auto &pair : rdm_map) {
      data.rdm1_operators.push_back(pair.second);
      rdm1_index_map[{pair.second.p, pair.second.q}] = current_idx++;
    }

    spdlog::info(
        "[Simulation] Successfully parsed {} operator groups for 1-RDM",
        data.rdm1_operators.size());

  } catch (const std::exception &e) {
    spdlog::error("[Simulation] Failed to generate/parse 1-RDM: {}", e.what());
    // Depending on rules, you could rethrow or continue without 1-RDM logging.
  }

  if (!ansatz.preserves_particle_number()) {
    std::string identity = std::string(physics.get_num_qubits(), 'I');
    int id_coeff = physics.get_num_qubits() / 2;
    std::vector<PauliStr> terms(physics.get_num_qubits() + 1);
    std::vector<qcomp> coeffs(physics.get_num_qubits() + 1, 0.0);

    terms[0] = getPauliStr(identity);
    coeffs[0] = id_coeff;

    for (int i = 0; i < physics.get_num_qubits(); ++i) {
      std::string term = identity;
      term[i] = 'Z';
      terms[i + 1] = getPauliStr(term);
      coeffs[i + 1] = -0.5;
    }
    data.number_op = createPauliStrSum(terms, coeffs);
    data.has_number_op = true;
  }

  optimizer.set_min_objective(cost_function, &data);

  // --- Allocating Diffraction Data ---
  int N_rdm = data.rdm1_operators.size();
  int n_orbs = physics.get_num_qubits() / 2;

  if (N_rdm > 0 && !fcalc_path.empty() && !ft_int_path.empty()) {
    spdlog::info("[Simulation] Loading diffraction data from {} and {}",
                 fcalc_path, ft_int_path);

    // 1. Read fcalc
    std::vector<double> exp_F;
    std::vector<double> exp_sigma;
    std::ifstream fcalc_file(fcalc_path);
    if (!fcalc_file.is_open()) {
      spdlog::error("Could not open fcalc file!");
    } else {
      std::string line;
      while (std::getline(fcalc_file, line)) {
        if (line.empty() || line[0] == '#')
          continue;
        std::stringstream ss(line);
        int h, k, l;
        double f_obs, sigma;
        if (ss >> h >> k >> l >> f_obs >> sigma) {
          exp_F.push_back(f_obs);
          exp_sigma.push_back(sigma);
        }
      }
    }

    int M = exp_F.size();
    spdlog::info("[Simulation] Parsed {} reflections from fcalc", M);

    data.exp_factors.resize(M);
    data.uncertainties.resize(M);
    for (int i = 0; i < M; ++i) {
      data.exp_factors(i) = std::complex<double>(exp_F[i], 0.0);
      data.uncertainties(i) = exp_sigma[i];
    }

    // 2. Read ft_int
    data.integrals = Eigen::MatrixXcd::Zero(M, N_rdm);
    std::ifstream ft_int_file(ft_int_path);
    if (!ft_int_file.is_open()) {
      spdlog::error("Could not open ft_int file!");
    } else {
      std::string line;
      int line_count = 0;
      int total_orbs_expected = n_orbs * n_orbs;

      while (std::getline(ft_int_file, line)) {
        if (line.empty() || line[0] == '#')
          continue;
        std::stringstream ss(line);
        int p_file, q_file;
        double real_val, imag_val;
        if (ss >> p_file >> q_file >> real_val >> imag_val) {
          int ref_idx = line_count / total_orbs_expected;
          if (ref_idx < M) {
            // Convert to 0-indexed values
            int p_0 = p_file - 1;
            int q_0 = q_file - 1;

            auto it = rdm1_index_map.find({p_0, q_0});
            if (it != rdm1_index_map.end()) {
              int rdm_idx = it->second;
              data.integrals(ref_idx, rdm_idx) =
                  std::complex<double>(real_val, imag_val);
            }
          }
          line_count++;
        }
      }
      spdlog::info("[Simulation] Loaded ft_int file ({} lines processed).",
                   line_count);
    }
  } else {
    spdlog::info(
        "[Simulation] No valid diffraction files specified or N_rdm == 0.");
  }

  double min_energy = 0.0;

  try {
    nlopt::result result = optimizer.optimize(optimal_params, min_energy);
    spdlog::info("[Simulation] Optimization finished successfully - Result "
                 "code: {}, Min Energy: {:.6f}",
                 (int)result, min_energy);
  } catch (const std::exception &e) {
    spdlog::error("[Simulation] NLopt optimization failed abruptly: {}",
                  e.what());
  }

  if (data.has_number_op) {
    destroyPauliStrSum(data.number_op);
  }

  // Properly destruct pre-allocated single-term Pauli sums to prevent memory
  // leaks
  for (auto &sum : data.single_term_sums) {
    destroyPauliStrSum(sum);
  }

  return min_energy;
}

//------------------------------------------------------------------------------
//     CONFIGURATION METHODS
//------------------------------------------------------------------------------

/**
 * @brief Sets the maximum number of function evaluations for the optimizer.
 * @param max_evals The maximum number of evaluations.
 */
void Simulation::set_max_evals(int max_evals) {
  optimizer.set_maxeval(max_evals);
}

/**
 * @brief Sets the relative tolerance for the optimizer convergence.
 * @param tol The relative tolerance value.
 */
void Simulation::set_tolerance(double tol) { optimizer.set_ftol_rel(tol); }

/**
 * @brief Sets the number of shots for noisy simulation.
 * @param shots Number of measurement shots (0 for exact simulation).
 */
void Simulation::set_shots(int shots) { n_shots = shots; }

/**
 * @brief Sets the scaling factor for the diffraction penalty.
 * @param lambda Scaling factor value.
 */
void Simulation::set_lambda(double lambda) { lambda_val = lambda; }

//------------------------------------------------------------------------------
//     STATISTICS & HELPER METHODS
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the variance of the energy estimate from the last run.
 * @return double The variance.
 */
double Simulation::get_last_variance() const { return last_variance; }

/**
 * @brief Retrieves the standard deviation of the energy estimate from the last
 * run.
 * @return double The standard deviation.
 */
double Simulation::get_last_std() const { return last_std; }

/**
 * @brief Calculates the probability distribution of the final state vector.
 *
 * Reconstructs the circuit with the given parameters and computes
 * probabilities. If shot noise is enabled, the probabilities are sampled from
 * the exact distribution.
 *
 * @param params The variational parameters to construct the circuit.
 * @return std::vector<double> The probability of each basis state.
 */
std::vector<double>
Simulation::get_probabilities(const std::vector<double> &params) {
  // Reset state and apply Hartree-Fock initialization
  initZeroState(qubits);
  int n_elec = physics.get_n_electrons();
  for (int i = 0; i < n_elec; ++i) {
    applyPauliX(qubits, i);
  }

  // Apply the Ansatz circuit
  ansatz.construct_circuit(qubits, params, physics.get_pauli_strings());

  // Calculate exact probabilities from state vector amplitudes
  int num_qubits = physics.get_num_qubits();
  long long dim = 1LL << num_qubits;
  std::vector<double> probs(dim);

  for (long long i = 0; i < dim; ++i) {
    qcomp amp = getQuregAmp(qubits, i);
    probs[i] = std::norm(std::complex<double>(amp.real(), amp.imag()));
  }

  // Apply shot noise sampling if requested
  if (n_shots > 0) {
    std::vector<double> sampled_probs(dim, 0.0);
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> d(probs.begin(), probs.end());

    for (int k = 0; k < n_shots; ++k) {
      int outcome = d(gen);
      sampled_probs[outcome] += 1.0;
    }

    for (auto &p : sampled_probs) {
      p /= n_shots;
    }
    return sampled_probs;
  }

  return probs;
}
