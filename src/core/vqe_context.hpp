//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file vqe_context.hpp
 * @author Rayan MALEK
 * @date 2026-05-05
 * @brief Header file for the VQEContext class, managing VQE data and resources.
 */

#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "ansatz.hpp"
#include "physics.hpp"
#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <nlohmann/json.hpp>
#include <vector>
#include <quest.h>

//------------------------------------------------------------------------------
//     STRUCTS
//------------------------------------------------------------------------------

/**
 * @struct RDM1Term
 * @brief Represents a single 1-Particle Reduced Density Matrix term $a^\dagger_p a_q$.
 *
 * It contains the mapped Pauli strings and their complex coefficients.
 */
struct RDM1Term {
  int p;
  int q;
  std::vector<PauliStr> strings;
  std::vector<qcomp> coeffs;
  
  PauliStrSum quest_sum_real;
  PauliStrSum quest_sum_imag;
  bool has_quest_sums = false;
};

/**
 * @struct RDM2Term
 * @brief Represents a single 2-Particle Reduced Density Matrix term $a^\dagger_p a^\dagger_q a_r a_s$.
 *
 * It contains the mapped Pauli strings and their complex coefficients.
 */
struct RDM2Term {
  int p;
  int q;
  int r;
  int s;
  std::vector<PauliStr> strings;
  std::vector<qcomp> coeffs;
};

//------------------------------------------------------------------------------
//     CLASS DECLARATION
//------------------------------------------------------------------------------

/**
 * @enum GradientMethod
 * @brief Enum for selecting the gradient evaluation method.
 */
enum class GradientMethod {
  FD,     // Finite Differences
  PSR,    // Parameter Shift Rule
  gPSR    // Generalized Parameter Shift Rule
};

/**
 * @class VQEContext
 * @brief Manages the data and resources required for a VQE evaluation.
 *
 * Replaces the old VQEData struct to properly manage QuEST resources using RAII.
 */
class VQEContext {
public:
  Ansatz &ansatz;           ///< Reference to the variational ansatz.
  Physics &physics;         ///< Reference to Physics object (needed for noise sim).

  Qureg qubits;             ///< The QuEST quantum register.
  PauliStrSum hamiltonian;  ///< The Hamiltonian structure.

  std::vector<std::string> paulis; ///< List of Pauli strings.

  ///< Callback function for real-time updates (iter, total_energy,
  ///< quantum_energy, chi_squared, probs, params).
  std::function<void(int, double, double, double, const std::vector<double> &,
                     const std::vector<double> &)>
      callback;

  GradientMethod grad_method = GradientMethod::PSR;
  double fd_tol = 1e-4;

  int current_iter = 0; ///< Current iteration counter.
  int n_electrons = 0;  ///< Number of electrons in the system.
  int num_qubits = 0;   ///< Number of qubits.
  int n_shots = 0;      ///< Number of shots for noise simulation.
  double lambda = 1.0;  ///< Scaling factor for diffraction penalty.

  double *variance_ptr = nullptr; ///< Pointer to store energy variance.
  double *std_ptr = nullptr;      ///< Pointer to store energy standard deviation.

  PauliStrSum number_penalty_op; ///< Penalty operator (N - N_target)^2 for particle number conservation.
  bool has_number_penalty = false; ///< Flag indicating if penalty operator is initialized.

  PauliStrSum spin_penalty_op; ///< Penalty operator (S_z - target_S_z)^2 for spin projection conservation.
  bool has_spin_penalty = false; ///< Flag indicating if spin penalty operator is initialized.

  std::vector<PauliStr> parsed_paulis; ///< Pre-parsed QuEST Pauli strings for noisy sim.
  std::vector<PauliStrSum> single_term_sums; ///< Pre-allocated PauliStrSums for each term.

  // 1-RDM terms
  std::vector<RDM1Term> rdm1_operators; ///< 1-RDM grouped Pauli strings and coefficients.
  std::vector<qcomp> current_1rdm;      ///< Latest evaluated 1-RDM expectation values.

  // Diffraction data (Eigen)
  Eigen::VectorXcd rdm1_alpha;   ///< Alpha spin-orbital components
  Eigen::VectorXcd rdm1_beta;    ///< Beta spin-orbital components
  Eigen::VectorXcd rdm1_spatial; ///< Total spatial 1-RDM vector

  Eigen::MatrixXcd integrals;    ///< Integral matrix for theoretically computed factors.
  Eigen::VectorXcd exp_factors;  ///< Experimental factors for comparison.
  Eigen::VectorXd uncertainties; ///< Experimental uncertainties.

  // Debug: last computed eta and structure factors
  double last_eta = 0.0;                 ///< Last computed scale factor η.
  Eigen::VectorXd last_calc_factors_abs; ///< Last computed |F_calc| array.

  bool is_setup = false;

  /**
   * @brief Constructs a VQEContext object and allocates the QuEST register.
   *
   * @param physics Reference to the Physics object.
   * @param ansatz Reference to the Ansatz object.
   */
  VQEContext(Physics &physics, Ansatz &ansatz);

  /**
   * @brief Destructor. Cleans up all QuEST resources.
   */
  ~VQEContext();

  /**
   * @brief Initializes the context data, generating Hamiltonians and parsing files.
   *
   * @param fcalc_path Path to the experimental factors file. Optional.
   * @param ft_int_path Path to the theoretical integrals file. Optional.
   */
  void setup(const std::string &fcalc_path = "", const std::string &ft_int_path = "");

private:
  void free_quest_resources();
};
