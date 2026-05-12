//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file simulation.hpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Header file for the Simulation class, managing VQE optimization.
 */

#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "ansatz.hpp"
#include "compat.h"
#include "opt.hpp"
#include "physics.hpp"
#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <nlohmann/json.hpp>
#include <nlopt.hpp>
#include <vector>

//------------------------------------------------------------------------------
//     STRUCTS
//------------------------------------------------------------------------------

#include "vqe_context.hpp"

//------------------------------------------------------------------------------
//     CLASS DECLARATION
//------------------------------------------------------------------------------

/**
 * @class Simulation
 * @brief Manages the Variational Quantum Eigensolver (VQE) simulation.
 *
 * This class orchestrates the VQE process, bridging the Physics model,
 * the Ansatz circuit, and the classical Optimizer (NLopt). It supports
 * both exact state-vector simulation and shot-noise simulation.
 */
class Simulation {
public:
  //----------------------------------------------------------------------------
  //     CONSTRUCTORS / DESTRUCTORS
  //----------------------------------------------------------------------------

  enum class OptType { NLOPT, SPSA };

  /**
   * @brief Constructs a Simulation object.
   *
   * @param context Reference to the VQEContext object.
   * @param algo The optimization algorithm to use (default: Nelder-Mead).
   */
  Simulation(VQEContext &context,
             nlopt::algorithm algo = nlopt::LN_NELDERMEAD);

  /**
   * @brief Destructor.
   */
  ~Simulation();

  //----------------------------------------------------------------------------
  //     CORE FUNCTIONALITY
  //----------------------------------------------------------------------------

  /**
   * @brief Runs the VQE optimization.
   *
   * @param optimal_params Output vector for the best found parameters.
   * @param callback Optional callback for iteration updates.
   * @return double The minimum energy found.
   */
  double run(std::vector<double> &optimal_params,
             std::function<void(int, double, double, double,
                                const std::vector<double> &,
                                const std::vector<double> &)>
                 callback = nullptr);

  //----------------------------------------------------------------------------
  //     CONFIGURATION
  //----------------------------------------------------------------------------

  /**
   * @brief Sets the maximum number of evaluations for the optimizer.
   * @param max_evals Max evaluations count.
   */
  void set_max_evals(int max_evals);

  /**
   * @brief Sets the relative tolerance for convergence.
   * @param tol Tolerance value (e.g., 1e-8).
   */
  void set_tolerance(double tol);

  /**
   * @brief Sets the optimizer type to use.
   * @param type OptType::NLOPT or OptType::SPSA.
   */
  void set_optimizer_type(OptType type);

  /**
   * @brief Sets the SPSA hyperparameters.
   * @param p SPSAParams struct.
   */
  void set_spsa_params(const SPSAParams &p);

  //----------------------------------------------------------------------------
  //     STATISTICS & RESULTS
  //----------------------------------------------------------------------------

  /**
   * @brief Gets the variance from the last energy evaluation.
   * @return double Energy variance.
   */
  double get_last_variance() const;

  /**
   * @brief Gets the standard deviation from the last energy evaluation.
   * @return double Energy standard deviation.
   */
  double get_last_std() const;

  /**
   * @brief Computes the probability distribution for a given set of parameters.
   * @param params Variational parameters.
   * @return std::vector<double> Vector of probabilities for basis states.
   */
  std::vector<double> get_probabilities(const std::vector<double> &params);

  /**
   * @brief Retrieves the final 1-RDM and 2-RDM evaluated matrices.
   * @return nlohmann::json JSON object containing 1-RDM and 2-RDM.
   */
  nlohmann::json get_rdms() const;

private:
  //----------------------------------------------------------------------------
  //     PRIVATE MEMBERS
  //----------------------------------------------------------------------------

  VQEContext &ctx;     ///< Reference to the VQE context.
  nlopt::opt optimizer; ///< Classical optimizer.
  SPSA_Optimizer spsa_optimizer; ///< SPSA optimizer.
  OptType opt_type_ = OptType::NLOPT; ///< Current optimizer type.

  double last_variance = 0.0; ///< Last computed variance.
  double last_std = 0.0;      ///< Last computed standard deviation.

  nlohmann::json final_rdms;  ///< Stored final 1-RDM and 2-RDM.

  //----------------------------------------------------------------------------
  //     INTERNAL METHODS
  //----------------------------------------------------------------------------

  /**
   * @brief Static cost function wrapper for NLopt.
   *
   * @param params Current parameters.
   * @param grad Gradient vector (if supported).
   * @param data Pointer to VQEContext struct.
   * @return double Energy value (to be minimized).
   */
  static double cost_function(const std::vector<double> &params,
                              std::vector<double> &grad, void *data);

  /**
   * @brief Helper function to evaluate functional (energy + 1-RDM) for a given
   * set of parameters.
   *
   * @param params Current parameters.
   * @param data Pointer to VQEContext struct.
   * @param local_qubits The quantum register to use for evaluation.
   * @param rdm1_out Vector to store the output 1-RDM evaluations.
   * @param out_quantum_energy Optional pointer to store extracted quantum
   * energy.
   * @param out_chi_squared Optional pointer to store extracted chi-squared
   * value.
   * @return double Calculated total energy value (including penalty).
   */
  static double evaluate_functional(const std::vector<double> &params,
                                    VQEContext *data, Qureg local_qubits,
                                    std::vector<qcomp> &rdm1_out,
                                    double *out_quantum_energy = nullptr,
                                    double *out_chi_squared = nullptr);
};
