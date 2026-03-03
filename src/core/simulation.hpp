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
#include "physics.hpp"
#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <nlopt.hpp>
#include <vector>

//------------------------------------------------------------------------------
//     STRUCTS
//------------------------------------------------------------------------------

/**
 * @struct RDM1Term
 * @brief Represents a single 1-Particle Reduced Density Matrix term
 * $a^\dagger_p a_q$.
 *
 * It contains the mapped Pauli strings and their complex coefficients.
 */
struct RDM1Term {
  int p;
  int q;
  std::vector<PauliStr> strings;
  std::vector<qcomp> coeffs;
};

/**
 * @struct VQEData
 * @brief Data structure passed to the cost function during optimization.
 *
 * Contains references to the system components (Ansatz, Physics, Qubits)
 * and runtime parameters required for energy evaluation and callback execution.
 */
struct VQEData {
  Ansatz &ansatz;           ///< Reference to the variational ansatz.
  Qureg &qubits;            ///< Reference to the QuEST quantum register.
  PauliStrSum &hamiltonian; ///< Reference to the Hamiltonian structure.

  const std::vector<std::string> &paulis; ///< List of Pauli strings.

  Physics &physics; ///< Reference to Physics object (needed for noise sim).

  ///< Callback function for real-time updates (iter, energy, probs, params).
  std::function<void(int, double, const std::vector<double> &,
                     const std::vector<double> &)>
      callback;

  int current_iter = 0; ///< Current iteration counter.
  int n_electrons = 0;  ///< Number of electrons in the system.
  int num_qubits = 0;   ///< Number of qubits.
  int n_shots = 0;      ///< Number of shots for noise simulation.

  double *variance_ptr = nullptr; ///< Pointer to store energy variance.
  double *std_ptr = nullptr; ///< Pointer to store energy standard deviation.

  PauliStrSum number_op; ///< Penalty operator for particle number conservation.
  bool has_number_op =
      false; ///< Flag indicating if penalty operator is initialized.

  std::vector<PauliStr>
      parsed_paulis; ///< Pre-parsed QuEST Pauli strings for noisy sim.
  std::vector<PauliStrSum>
      single_term_sums; ///< Pre-allocated PauliStrSums for each term.

  // 1-RDM terms
  std::vector<RDM1Term>
      rdm1_operators; ///< 1-RDM grouped Pauli strings and coefficients.
  std::vector<qcomp>
      current_1rdm; ///< Latest evaluated 1-RDM expectation values.

  // Diffraction data (Eigen)
  Eigen::MatrixXcd
      integrals; ///< Integral matrix for theoretically computed factors.
  Eigen::VectorXcd exp_factors;  ///< Experimental factors for comparison.
  Eigen::VectorXd uncertainties; ///< Experimental uncertainties.
};

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

  /**
   * @brief Constructs a Simulation object.
   *
   * @param physics Reference to the Physics object.
   * @param ansatz Reference to the Ansatz object.
   * @param algo The optimization algorithm to use (default: Nelder-Mead).
   */
  Simulation(Physics &physics, Ansatz &ansatz,
             nlopt::algorithm algo = nlopt::LN_NELDERMEAD);

  /**
   * @brief Destructor. Cleans up quantum resources.
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
   * @param fcalc_path Path to the experimental factors file.
   * @param ft_int_path Path to the theoretical integrals file.
   * @return double The minimum energy found.
   */
  double run(std::vector<double> &optimal_params,
             std::function<void(int, double, const std::vector<double> &,
                                const std::vector<double> &)>
                 callback = nullptr,
             const std::string &fcalc_path = "",
             const std::string &ft_int_path = "");

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
   * @brief Sets the number of shots for noisy simulation.
   * @param shots Number of shots (0 for exact simulation).
   */
  void set_shots(int shots);

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

private:
  //----------------------------------------------------------------------------
  //     PRIVATE MEMBERS
  //----------------------------------------------------------------------------

  Physics &physics;     ///< System physics model.
  Ansatz &ansatz;       ///< Quantum circuit ansatz.
  Qureg qubits;         ///< QuEST quantum register.
  nlopt::opt optimizer; ///< Classical optimizer.

  int n_shots = 0;            ///< configured number of shots.
  double last_variance = 0.0; ///< Last computed variance.
  double last_std = 0.0;      ///< Last computed standard deviation.

  //----------------------------------------------------------------------------
  //     INTERNAL METHODS
  //----------------------------------------------------------------------------

  /**
   * @brief Static cost function wrapper for NLopt.
   *
   * @param params Current parameters.
   * @param grad Gradient vector (if supported).
   * @param data Pointer to VQEData struct.
   * @return double Energy value (to be minimized).
   */
  static double cost_function(const std::vector<double> &params,
                              std::vector<double> &grad, void *data);

  /**
   * @brief Helper function to evaluate functional (energy + 1-RDM) for a given
   * set of parameters.
   *
   * @param params Current parameters.
   * @param data Pointer to VQEData struct.
   * @param local_qubits The quantum register to use for evaluation.
   * @param rdm1_out Vector to store the output 1-RDM evaluations.
   * @return double Calculated energy value.
   */
  static double evaluate_functional(const std::vector<double> &params,
                                    VQEData *data, Qureg local_qubits,
                                    std::vector<qcomp> &rdm1_out);
};
