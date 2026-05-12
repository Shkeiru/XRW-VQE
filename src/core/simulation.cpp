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
#include <bitset>
#include <cmath>
#include <complex>
#include <fstream> // Added by user instruction
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp> // Added by user instruction
#include <omp.h>
#include <quest.h>
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
 * Configures the optimizer with default bounds and tolerance.
 *
 * @param context Reference to the VQEContext object.
 * @param algo The NLopt optimization algorithm to use.
 */
Simulation::Simulation(VQEContext &context, nlopt::algorithm algo)
    : ctx(context),
      optimizer(algo, context.ansatz.get_num_params()),
      spsa_optimizer(context.ansatz.get_num_params()) {

  // Configure the non-linear optimizer parameters
  optimizer.set_ftol_rel(1e-8);
  spsa_optimizer.set_ftol_rel(1e-8);
}

/**
 * @brief Destructor.
 */
Simulation::~Simulation() {}

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
                                       VQEContext *data, Qureg local_qubits,
                                       std::vector<qcomp> &rdm1_out,
                                       double *out_quantum_energy,
                                       double *out_chi_squared) {

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

  // Apply variance penalty for deviations from expected electron number
  if (!data->ansatz.preserves_particle_number() && data->has_number_penalty) {
    qreal penalty_val = calcExpecPauliStrSum(local_qubits, data->number_penalty_op);
    energy += 14 * penalty_val;
  }

  if (!data->ansatz.preserves_spin() && data->has_spin_penalty) {
    // 1. Calcul de <S^2>
    qreal s2_val = calcExpecPauliStrSum(local_qubits, data->spin_penalty_op);
    
    // 2. Détermination de la cible S(S+1)
    double target_s = data->physics.get_target_spin() / 2.0; 
    double target_s2 = target_s * (target_s + 1.0);
    
    // 3. Pénalité quadratique (mean-field penalty)
    energy += 14 * std::pow(s2_val - target_s2, 2);
  }

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

  if (out_quantum_energy)
    *out_quantum_energy = energy;
  if (out_chi_squared)
    *out_chi_squared = 0.0;

  if (!data->current_1rdm.empty()) {
    int n_orbs = data->num_qubits / 2;
    data->rdm1_alpha.resize(n_orbs * n_orbs);
    data->rdm1_beta.resize(n_orbs * n_orbs);
    data->rdm1_spatial.resize(n_orbs * n_orbs);
    data->rdm1_alpha.setZero();
    data->rdm1_beta.setZero();

    // Sum over alpha and beta components
    for (size_t idx = 0; idx < data->rdm1_operators.size(); ++idx) {
      int p = data->rdm1_operators[idx].p;
      int q = data->rdm1_operators[idx].q;
      int p_spatial = (n_orbs-1) - p / 2;
      int q_spatial = (n_orbs-1) - q / 2;
      int spatial_idx = p_spatial * n_orbs + q_spatial;

      if (p % 2 == 0 && q % 2 == 0) {
        data->rdm1_alpha(spatial_idx) += rdm1_out[idx];
      } else if (p % 2 == 1 && q % 2 == 1) {
        data->rdm1_beta(spatial_idx) += rdm1_out[idx];
      }
    }

    data->rdm1_spatial = data->rdm1_alpha + data->rdm1_beta;

    if (data->integrals.size() > 0) {
      // Perform the matrix-vector multiplication to get the theoretical factors
      Eigen::VectorXcd calc_factors = data->integrals * data->rdm1_spatial;

      // Computation of eta
      double eta = (calc_factors.array().abs() * data->exp_factors.array().abs() / data->uncertainties.array().abs2()).sum() /
             (calc_factors.array().abs2() / data->uncertainties.array().abs2()).sum();

      // Store for debug output
      data->last_eta = eta;
      data->last_calc_factors_abs = calc_factors.cwiseAbs().cast<double>();

      double chi_squared =
          (1.0 / data->exp_factors.size()) *
          (eta * calc_factors.cwiseAbs() - data->exp_factors.cwiseAbs())
              .cwiseAbs2()
              .cwiseQuotient(data->uncertainties.cwiseAbs2())
              .sum();

      if (out_chi_squared)
        *out_chi_squared = chi_squared;
      energy += data->lambda * chi_squared;
    }
  }

  return energy;
}

double Simulation::cost_function(const std::vector<double> &params,
                                 std::vector<double> &grad, void *data_ptr) {

  VQEContext *data = static_cast<VQEContext *>(data_ptr);

  double current_quantum_energy = 0.0;
  double current_chi_squared = 0.0;

  // 1. Calculate the energy of the current point on the main register
  double base_energy = evaluate_functional(params, data, data->qubits, data->current_1rdm, &current_quantum_energy, &current_chi_squared);

  // 2. Calculate the local gradients using Parameter Shift Rule (PSR)
  // ONLY if requested by the optimizer
  if (!grad.empty()) {
    int num_params = params.size();

    // Initialize Qureg
    Qureg local_qubits = createQureg(data->num_qubits);

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

    data->callback(data->current_iter, base_energy, current_quantum_energy,
                   current_chi_squared, probs, params);
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
 * @param callback Function to be called at each iteration (step, energy, probabilities).
 * @return double The minimum energy found after optimization.
 */
double Simulation::run(
    std::vector<double> &optimal_params,
    std::function<void(int, double, double, double, const std::vector<double> &,
                       const std::vector<double> &)>
        callback) {

  // Initialize VQE data context for the cost function
  spdlog::info("[Simulation] Starting VQE Optimization with {} params and max {} evaluations",
               ctx.ansatz.get_num_params(), optimizer.get_maxeval());

  ctx.callback = callback;
  ctx.current_iter = 0;
  ctx.variance_ptr = &last_variance;
  ctx.std_ptr = &last_std;

  // Reset statistical trackers
  last_variance = 0.0;
  last_std = 0.0;

  optimizer.set_min_objective(cost_function, &ctx);
  spsa_optimizer.set_min_objective(cost_function, &ctx);

  double min_energy = 0.0;

  try {
    nlopt::result result;
    if (opt_type_ == OptType::NLOPT) {
      result = optimizer.optimize(optimal_params, min_energy);
    } else {
      result = spsa_optimizer.optimize(optimal_params, min_energy);
    }
    spdlog::info("[Simulation] Optimization finished successfully - Result code: {}, Min Energy: {:.6f}",
                 (int)result, min_energy);
  } catch (const std::exception &e) {
    spdlog::error("[Simulation] Optimization failed abruptly: {}", e.what());
  }

  //----------------------------------------------------------------------------
  // Final Evaluation of 1-RDM and 2-RDM with optimal parameters
  //----------------------------------------------------------------------------
  spdlog::info("[Simulation] Evaluating final 1-RDM and 2-RDM...");

  std::vector<qcomp> final_1rdm_vals;
  evaluate_functional(optimal_params, &ctx, ctx.qubits, final_1rdm_vals);

  nlohmann::json rdm_output;
  nlohmann::json rdm1_json = nlohmann::json::array();

  for (size_t i = 0; i < ctx.rdm1_operators.size(); ++i) {
    nlohmann::json term_json;
    term_json["p"] = ctx.rdm1_operators[i].p;
    term_json["q"] = ctx.rdm1_operators[i].q;
    term_json["val_real"] = final_1rdm_vals[i].real();
    term_json["val_imag"] = final_1rdm_vals[i].imag();
    rdm1_json.push_back(term_json);
  }
  rdm_output["1-RDM"] = rdm1_json;

  // 2-RDM Parsing
  std::vector<RDM2Term> rdm2_operators;
  try {
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 python/generate_2rdm.py --n_qubits " + std::to_string(ctx.physics.get_num_qubits()) + " --mapping jordan_wigner";
    spdlog::info("[Simulation] Generating 2-RDM mapping: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (pipe) {
      char buffer[256];
      std::string json_filepath_2rdm = "";
      while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        json_filepath_2rdm += buffer;
      }
      _pclose(pipe);

      json_filepath_2rdm.erase(std::remove(json_filepath_2rdm.begin(), json_filepath_2rdm.end(), '\n'), json_filepath_2rdm.end());
      json_filepath_2rdm.erase(std::remove(json_filepath_2rdm.begin(), json_filepath_2rdm.end(), '\r'), json_filepath_2rdm.end());

      std::ifstream f2(json_filepath_2rdm);
      if (f2.is_open()) {
        nlohmann::json rdm2_json_file;
        f2 >> rdm2_json_file;

        std::map<std::tuple<int, int, int, int>, RDM2Term> rdm2_map;
        for (const auto &item : rdm2_json_file) {
          int p = item["p"];
          int q = item["q"];
          int r = item["r"];
          int s = item["s"];
          double c_real = item["coeff_real"];
          double c_imag = item["coeff_imag"];
          std::string pauli_string = item["string"];

          std::tuple<int, int, int, int> pqrs = {p, q, r, s};
          if (rdm2_map.find(pqrs) == rdm2_map.end()) {
            rdm2_map[pqrs] = {p, q, r, s, {}, {}};
          }
          rdm2_map[pqrs].strings.push_back(getPauliStr(pauli_string));
          rdm2_map[pqrs].coeffs.push_back({c_real, c_imag});
        }
        for (auto &pair : rdm2_map) {
          rdm2_operators.push_back(pair.second);
        }
        spdlog::info("[Simulation] Successfully parsed {} operator groups for 2-RDM", rdm2_operators.size());
      }
    }
  } catch (const std::exception &e) {
    spdlog::error("[Simulation] Failed to generate/parse 2-RDM: {}", e.what());
  }

  // 2-RDM Evaluation
  nlohmann::json rdm2_json = nlohmann::json::array();
  for (size_t idx = 0; idx < rdm2_operators.size(); ++idx) {
    const auto &rdm_term = rdm2_operators[idx];
    qcomp term_expectation = 0.0;

    for (size_t s = 0; s < rdm_term.strings.size(); ++s) {
      if (std::abs(rdm_term.coeffs[s].real()) < 1e-9 && std::abs(rdm_term.coeffs[s].imag()) < 1e-9) {
        continue;
      }
      qcomp one = 1.0;
      PauliStr p_arr[] = {rdm_term.strings[s]};
      PauliStrSum temp_sum = createPauliStrSum(p_arr, &one, 1);
      
      double p_expect_real = calcExpecPauliStrSum(ctx.qubits, temp_sum);
      destroyPauliStrSum(temp_sum);

      term_expectation += rdm_term.coeffs[s] * p_expect_real;
    }

    nlohmann::json term_json;
    term_json["p"] = rdm_term.p;
    term_json["q"] = rdm_term.q;
    term_json["r"] = rdm_term.r;
    term_json["s"] = rdm_term.s;
    term_json["val_real"] = term_expectation.real();
    term_json["val_imag"] = term_expectation.imag();
    rdm2_json.push_back(term_json);
  }
  rdm_output["2-RDM"] = rdm2_json;
  
  // 1-RDM Spatiale
  nlohmann::json rdm1_spat_json = nlohmann::json::array();
  int spat_n_orbs = ctx.num_qubits / 2;
  for (int p = 0; p < spat_n_orbs; ++p) {
    for (int q = 0; q < spat_n_orbs; ++q) {
      int spatial_idx = p * spat_n_orbs + q;
      nlohmann::json trm;
      trm["p"] = p;
      trm["q"] = q;
      if (ctx.rdm1_spatial.size() > spatial_idx) {
        trm["val_real"] = ctx.rdm1_spatial(spatial_idx).real();
        trm["val_imag"] = ctx.rdm1_spatial(spatial_idx).imag();
      } else {
        trm["val_real"] = 0.0;
        trm["val_imag"] = 0.0;
      }
      rdm1_spat_json.push_back(trm);
    }
  }
  rdm_output["1-RDM_spatial"] = rdm1_spat_json;

  // Vector State direct from the QuEST simulation
  long long dim = 1LL << ctx.num_qubits;
  nlohmann::json state_json = nlohmann::json::array();
  for (long long k = 0; k < dim; ++k) {
    qcomp amp = getQuregAmp(ctx.qubits, k);
    nlohmann::json amp_j;
    amp_j["real"] = amp.real();
    amp_j["imag"] = amp.imag();
    state_json.push_back(amp_j);
  }
  rdm_output["state_vector"] = state_json;

  // Debug: final eta and structure factors
  rdm_output["final_eta"] = ctx.last_eta;
  {
    nlohmann::json cf_json = nlohmann::json::array();
    for (int i = 0; i < ctx.last_calc_factors_abs.size(); ++i) {
      cf_json.push_back(ctx.last_calc_factors_abs(i));
    }
    rdm_output["final_calc_factors"] = cf_json;
  }

  final_rdms = rdm_output;

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
  spsa_optimizer.set_maxeval(max_evals);
}

/**
 * @brief Sets the relative tolerance for the optimizer convergence.
 * @param tol The relative tolerance value.
 */
void Simulation::set_tolerance(double tol) { 
  optimizer.set_ftol_rel(tol); 
  spsa_optimizer.set_ftol_rel(tol);
}





/**
 * @brief Sets the optimizer type to use.
 * @param type OptType::NLOPT or OptType::SPSA.
 */
void Simulation::set_optimizer_type(OptType type) { opt_type_ = type; }

/**
 * @brief Sets the SPSA hyperparameters.
 * @param p SPSAParams struct.
 */
void Simulation::set_spsa_params(const SPSAParams &p) { spsa_optimizer.set_spsa_params(p); }



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
  int num_qubits = ctx.num_qubits;
  long long dim = 1LL << num_qubits;
  std::vector<double> probs(dim);

  Qureg local_qubits = createQureg(num_qubits);
  initZeroState(local_qubits);
  ctx.ansatz.construct_circuit(local_qubits, params, ctx.paulis);

  for (long long i = 0; i < dim; ++i) {
    qcomp amp = getQuregAmp(local_qubits, i);
    probs[i] = std::norm(std::complex<double>(amp.real(), amp.imag()));
  }

  if (ctx.n_shots > 0) {
    std::vector<double> sampled_probs(dim, 0.0);
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> d(probs.begin(), probs.end());

    for (int k = 0; k < ctx.n_shots; ++k) {
      int outcome = d(gen);
      sampled_probs[outcome] += 1.0;
    }

    for (auto &p : sampled_probs) {
      p /= ctx.n_shots;
    }
    destroyQureg(local_qubits);
    return sampled_probs;
  }

  destroyQureg(local_qubits);
  return probs;
}

nlohmann::json Simulation::get_rdms() const {
  return final_rdms;
}
