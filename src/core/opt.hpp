//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file opt.hpp
 * @author Rayan MALEK
 * @date 2026-03-16
 * @brief SPSA optimizer — drop-in complement to NLopt for gradient-free
 *        optimization of the VQE cost function.
 */

#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include <functional>
#include <nlopt.hpp>
#include <random>
#include <vector>

//------------------------------------------------------------------------------
//     STRUCTS
//------------------------------------------------------------------------------

/**
 * @struct SPSAParams
 * @brief Hyperparameters for the SPSA algorithm (Spall 1998).
 *
 * Gain sequences:
 *   aₖ = a / (A + k)^alpha   — step size
 *   cₖ = c / k^gamma         — perturbation amplitude
 *
 * Recommended defaults (Spall 1998):
 *   alpha = 0.602, gamma = 0.101
 *   A ≈ 10% of max_evals
 */
struct SPSAParams {
  double a     = 0.1;    ///< Step size numerator.
  double c     = 0.1;    ///< Perturbation amplitude numerator.
  double A     = 10.0;   ///< Stability constant (set to ~10% of max_evals).
  double alpha = 0.602;  ///< Step size decay exponent.
  double gamma = 0.101;  ///< Perturbation decay exponent.
};

/**
 * @struct SPSAResult
 * @brief Result returned by spsa_optimize.
 */
struct SPSAResult {
  nlopt::result status;  ///< Mirrors NLopt result codes for compatibility.
  double        minval;  ///< Best function value found.
  int           n_evals; ///< Total number of cost-function evaluations.
  int           n_iters; ///< Number of SPSA steps performed.
};

//------------------------------------------------------------------------------
//     GAIN SEQUENCES (Spall 1998)
//------------------------------------------------------------------------------

/**
 * @brief Step size gain: aₖ = a / (A + k)^alpha.  k is 1-based.
 */
inline double spsa_gain_a(const SPSAParams &p, int k)
{
    return p.a / std::pow(p.A + static_cast<double>(k), p.alpha);
}

/**
 * @brief Perturbation gain: cₖ = c / k^gamma.  k is 1-based.
 */
inline double spsa_gain_c(const SPSAParams &p, int k)
{
    return p.c / std::pow(static_cast<double>(k), p.gamma);
}

//------------------------------------------------------------------------------
//     CLASS SPSA_Optimizer
//------------------------------------------------------------------------------

/**
 * @class SPSA_Optimizer
 * @brief A drop-in replacement for nlopt::opt used to run the SPSA algorithm.
 * 
 * Mirrors the C++ API of nlopt::opt to allow seamless integration into Simulation.
 */
class SPSA_Optimizer {
public:
  // Signature matching nlopt::vfunc
  using vfunc = double (*)(const std::vector<double> &x, std::vector<double> &grad, void *data);

  /// Lightweight evaluation function (no gradient, no callback).
  using eval_fn_t = std::function<double(const std::vector<double> &)>;

  /**
   * @brief Construct a new SPSA Optimizer.
   * @param dim Dimension of the parameter space.
   */
  SPSA_Optimizer(int dim);

  /**
   * @brief Set the objective function to minimize.
   * @param f Function pointer matching nlopt signature.
   * @param f_data Pointer to user data context.
   */
  void set_min_objective(vfunc f, void *f_data);

  /**
   * @brief Set the maximum number of function evaluations.
   * @param m Max evaluations.
   */
  void set_maxeval(int m);

  /**
   * @brief Get the configured maximum evaluations.
   * @return int 
   */
  int get_maxeval() const;

  /**
   * @brief Set the relative tolerance for stopping criterion.
   * @param tol Tolerance.
   */
  void set_ftol_rel(double tol);

  /**
   * @brief Set custom SPSA hyperparameters.
   * @param p SPSA parameters struct.
   */
  void set_spsa_params(const SPSAParams &p);

  /**
   * @brief Set a lightweight evaluation function for internal perturbation evals.
   *
   * This function is called for f(x+cΔ) and f(x-cΔ) without triggering the
   * cost_function callback. The full cost_function (set via set_min_objective)
   * is called once per step after the parameter update.
   *
   * @param fn Evaluation function: params -> energy.
   */
  void set_eval_function(eval_fn_t fn);

  /**
   * @brief Run optimization.
   * @param x In/Out parameter vector.
   * @param minf Output minimum function value.
   * @return nlopt::result Result code mimicking NLopt.
   */
  nlopt::result optimize(std::vector<double> &x, double &minf);

private:
  int dim_;
  vfunc f_ = nullptr;
  void *f_data_ = nullptr;
  eval_fn_t eval_fn_ = nullptr;
  int max_evals_ = 1000;
  double ftol_rel_ = 1e-8;
  SPSAParams params_;
};

//------------------------------------------------------------------------------
//     PUBLIC API (Legacy SPSA functional interface)
//------------------------------------------------------------------------------

/**
 * @brief Run SPSA minimisation on an arbitrary scalar objective.
 *
 * The objective must be callable as  f(params) -> double.
 * It is evaluated exactly twice per iteration — never more.
 *
 * @param f          Objective function to minimise.
 * @param x          In:  initial point.  Out: best point found.
 * @param lb         Lower bounds (same size as x, or empty → unbounded).
 * @param ub         Upper bounds (same size as x, or empty → unbounded).
 * @param p          SPSA hyperparameters (use defaults for a first run).
 * @param max_evals  Budget: stop after this many calls to f.
 * @param ftol_rel   Stop when |Δf|/|f| < ftol_rel over one step (0 = off).
 * @param seed       RNG seed for reproducibility (0 = random).
 * @return SPSAResult
 */
SPSAResult spsa_optimize(
    std::function<double(const std::vector<double> &)> f,
    std::vector<double>                               &x,
    const std::vector<double>                         &lb,
    const std::vector<double>                         &ub,
    const SPSAParams                                  &p         = SPSAParams{},
    int                                                max_evals = 1000,
    double                                             ftol_rel  = 1e-8,
    unsigned int                                       seed      = 0);