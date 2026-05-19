//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file opt.cpp
 * @author Rayan MALEK
 * @date 2026-03-16
 * @brief SPSA optimizer implementation.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "opt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

//------------------------------------------------------------------------------
//     HELPERS
//------------------------------------------------------------------------------

namespace {

/**
 * @brief Clip a value into [lo, hi].  hi < lo is treated as unbounded.
 */
inline double clip(double v, double lo, double hi)
{
    if (lo <= hi) {
        v = std::max(v, lo);
        v = std::min(v, hi);
    }
    return v;
}

/**
 * @brief Apply per-component box constraints to x.
 */
void apply_bounds(std::vector<double>       &x,
                  const std::vector<double> &lb,
                  const std::vector<double> &ub)
{
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double lo = lb.empty() ? -std::numeric_limits<double>::max() : lb[i];
        const double hi = ub.empty() ?  std::numeric_limits<double>::max() : ub[i];
        x[i] = clip(x[i], lo, hi);
    }
}

} // anonymous namespace

// Use the header-exposed gain functions throughout this file.
static inline double gain_a(const SPSAParams &p, int k) { return spsa_gain_a(p, k); }
static inline double gain_c(const SPSAParams &p, int k) { return spsa_gain_c(p, k); }

//------------------------------------------------------------------------------
//     IMPLEMENTATION
//------------------------------------------------------------------------------

SPSAResult spsa_optimize(
    std::function<double(const std::vector<double> &)> f,
    std::vector<double>                               &x,
    const std::vector<double>                         &lb,
    const std::vector<double>                         &ub,
    const SPSAParams                                  &p,
    int                                                max_evals,
    double                                             ftol_rel,
    unsigned int                                       seed)
{
    // ------------------------------------------------------------------ setup
    const std::size_t n = x.size();

    if (!lb.empty() && lb.size() != n)
        throw std::invalid_argument("spsa_optimize: lb size mismatch");
    if (!ub.empty() && ub.size() != n)
        throw std::invalid_argument("spsa_optimize: ub size mismatch");
    if (max_evals < 2)
        throw std::invalid_argument("spsa_optimize: max_evals must be >= 2");

    // RNG — Bernoulli ±1
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::bernoulli_distribution coin(0.5);

    // Working vectors
    std::vector<double> x_plus(n), x_minus(n), delta(n);
    std::vector<double> x_best = x;

    // Clip starting point into bounds
    apply_bounds(x, lb, ub);

    // Evaluate at initial point (counts toward budget)
    double f_curr  = f(x);
    double f_best  = f_curr;
    int    n_evals = 1;
    int    n_iters = 0;

    // ------------------------------------------------------------------ loop
    while (n_evals + 2 <= max_evals) {

        ++n_iters;
        const int    k  = n_iters;           // 1-based iteration index
        const double ak = gain_a(p, k);
        const double ck = gain_c(p, k);

        // --- Draw Bernoulli ±1 perturbation vector
        for (std::size_t i = 0; i < n; ++i)
            delta[i] = coin(rng) ? 1.0 : -1.0;

        // --- Build x± and apply bounds
        for (std::size_t i = 0; i < n; ++i) {
            x_plus[i]  = x[i] + ck * delta[i];
            x_minus[i] = x[i] - ck * delta[i];
        }
        apply_bounds(x_plus,  lb, ub);
        apply_bounds(x_minus, lb, ub);

        // --- Two function evaluations  (this is the entire cost of one step)
        const double f_plus  = f(x_plus);
        const double f_minus = f(x_minus);
        n_evals += 2;

        // --- Simultaneous gradient approximation
        //     ĝᵢ = (f+ − f−) / (2 cₖ Δᵢ)
        //     Update: xᵢ ← xᵢ − aₖ ĝᵢ
        const double diff = f_plus - f_minus;
        for (std::size_t i = 0; i < n; ++i)
            x[i] -= ak * diff / (2.0 * ck * delta[i]);

        apply_bounds(x, lb, ub);

        // --- Track best point seen
        //     We re-use f_plus / f_minus to avoid an extra evaluation.
        //     The updated x is not yet evaluated — record best among ±.
        if (f_plus < f_best) { f_best = f_plus;  x_best = x_plus;  }
        if (f_minus < f_best){ f_best = f_minus; x_best = x_minus; }

        // --- ftol_rel stopping criterion
        //     Compare the average of the two bracket values to f_curr.
        const double f_avg = 0.5 * (f_plus + f_minus);
        if (ftol_rel > 0.0 && std::abs(f_curr) > 0.0) {
            if (std::abs(f_avg - f_curr) / std::abs(f_curr) < ftol_rel)
                break;
        }
        f_curr = f_avg;
    }

    // Return best known point (not necessarily the last iterate)
    x = x_best;

    SPSAResult result;
    result.minval  = f_best;
    result.n_evals = n_evals;
    result.n_iters = n_iters;
    result.status  = (n_evals >= max_evals)
                         ? nlopt::MAXEVAL_REACHED
                         : nlopt::FTOL_REACHED;
    return result;
}

//------------------------------------------------------------------------------
//     CLASS SPSA_Optimizer IMPLEMENTATION
//------------------------------------------------------------------------------

SPSA_Optimizer::SPSA_Optimizer(int dim) : dim_(dim) {}

void SPSA_Optimizer::set_min_objective(vfunc f, void *f_data) {
  f_ = f;
  f_data_ = f_data;
}

void SPSA_Optimizer::set_maxeval(int m) {
  max_evals_ = m;
}

int SPSA_Optimizer::get_maxeval() const {
  return max_evals_;
}

void SPSA_Optimizer::set_ftol_rel(double tol) {
  ftol_rel_ = tol;
}

void SPSA_Optimizer::set_spsa_params(const SPSAParams &p) {
  params_ = p;
}

void SPSA_Optimizer::set_eval_function(eval_fn_t fn) {
  eval_fn_ = std::move(fn);
}

nlopt::result SPSA_Optimizer::optimize(std::vector<double> &x, double &minf) {
  if (!f_) {
    throw std::runtime_error("SPSA_Optimizer: objective function not set.");
  }
  if (!eval_fn_) {
    throw std::runtime_error("SPSA_Optimizer: eval function not set. "
                             "Call set_eval_function() before optimize().");
  }

  const std::size_t n = static_cast<std::size_t>(dim_);

  // RNG — Bernoulli ±1
  std::mt19937 rng(std::random_device{}());
  std::bernoulli_distribution coin(0.5);

  // Working vectors
  std::vector<double> x_plus(n), x_minus(n), delta(n);
  std::vector<double> x_best = x;
  std::vector<double> dummy_grad;

  // Evaluate initial point via cost_function (fires callback)
  double f_curr = f_(x, dummy_grad, f_data_);
  double f_prev = f_curr;
  double f_best = f_curr;
  int    n_iters = 0;

  // ------------------------------------------------------------------ loop
  // max_evals_ is now the number of STEPS, not raw function evaluations.
  while (n_iters < max_evals_) {

    ++n_iters;
    const int    k  = n_iters;
    const double ak = gain_a(params_, k);
    const double ck = gain_c(params_, k);

    // --- Draw Bernoulli ±1 perturbation vector
    for (std::size_t i = 0; i < n; ++i)
      delta[i] = coin(rng) ? 1.0 : -1.0;

    // --- Build x±
    for (std::size_t i = 0; i < n; ++i) {
      x_plus[i]  = x[i] + ck * delta[i];
      x_minus[i] = x[i] - ck * delta[i];
    }

    // --- Two internal evaluations via eval_fn_ (NO callback)
    const double f_plus  = eval_fn_(x_plus);
    const double f_minus = eval_fn_(x_minus);

    // --- Simultaneous gradient approximation + parameter update
    const double diff = f_plus - f_minus;
    for (std::size_t i = 0; i < n; ++i)
      x[i] -= ak * diff / (2.0 * ck * delta[i]);

    // --- Evaluate at updated point via cost_function (fires callback)
    f_curr = f_(x, dummy_grad, f_data_);

    // --- Track best point seen
    if (f_curr < f_best) { f_best = f_curr;  x_best = x; }
    if (f_plus < f_best) { f_best = f_plus;  x_best = x_plus; }
    if (f_minus < f_best){ f_best = f_minus; x_best = x_minus; }

    // --- ftol_rel stopping criterion
    if (ftol_rel_ > 0.0 && std::abs(f_prev) > 0.0) {
      if (std::abs(f_curr - f_prev) / std::abs(f_prev) < ftol_rel_)
        break;
    }
    f_prev = f_curr;
  }

  // Return best known point
  x = x_best;
  minf = f_best;

  return (n_iters >= max_evals_) ? nlopt::MAXEVAL_REACHED
                                : nlopt::FTOL_REACHED;
}