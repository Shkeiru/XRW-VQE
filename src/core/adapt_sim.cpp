#include "adapt_sim.hpp"
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

//------------------------------------------------------------------------------
//     OPERATOR POOL
//------------------------------------------------------------------------------

std::vector<std::vector<GadgetInst>> OperatorPool::load_from_json(const std::string& filepath) {
  std::vector<std::vector<GadgetInst>> pool;
  std::ifstream f(filepath);
  if (!f.good()) {
    spdlog::error("[OperatorPool] Could not find pool file at {}", filepath);
    return pool;
  }
  try {
    json j;
    f >> j;
    for (const auto& op_json : j["pool"]) {
      std::vector<GadgetInst> op;
      for (const auto& gadget_json : op_json["gadgets"]) {
        GadgetInst inst;
        inst.pauli_chars = gadget_json["pauli_chars"].get<std::string>();
        inst.targets = gadget_json["targets"].get<std::vector<int>>();
        inst.multiplier = gadget_json["multiplier"].get<double>();
        inst.param_idx = 0; // Will be relinked by ADAPTAnsatz
        op.push_back(inst);
      }
      pool.push_back(op);
    }
    spdlog::info("[OperatorPool] Loaded {} operators from {}", pool.size(), filepath);
  } catch (const std::exception& e) {
    spdlog::error("[OperatorPool] Error parsing JSON: {}", e.what());
  }
  return pool;
}

//------------------------------------------------------------------------------
//     ADAPT SIMULATOR
//------------------------------------------------------------------------------

ADAPT_sim::ADAPT_sim(Physics& physics,
                     std::string optimizer_name, int max_evals, double vqe_tol, double adapt_epsilon,
                     int n_shots, double lambda_val,
                     std::string fcalc_path, std::string ft_int_path, std::string output_json_path,
                     int max_macro_iter,
                     std::function<void(int, double, double, double, const std::vector<double>&, const std::vector<double>&)> callback)
    : physics(physics),
      optimizer_name(optimizer_name), max_evals(max_evals), vqe_tol(vqe_tol), adapt_epsilon(adapt_epsilon),
      n_shots(n_shots), lambda_val(lambda_val), fcalc_path(fcalc_path), ft_int_path(ft_int_path),
      output_json_path(output_json_path), max_macro_iter(max_macro_iter), callback(callback) {}

void ADAPT_sim::run_adapt() {
  spdlog::info("[ADAPT_sim] Starting ADAPT-VQE with epsilon={}", adapt_epsilon);

  // 1. Create ADAPT Ansatz
  ADAPTAnsatz ansatz(physics.get_num_qubits(), physics.get_n_electrons());

  // 2. Initialize VQEContext ONLY ONCE
  VQEContext ctx(physics, ansatz);
  ctx.n_shots = n_shots;
  ctx.lambda = lambda_val;
  ctx.setup(fcalc_path, ft_int_path);

  // 3. Load Operator Pool
  // We need to call generate_pool.py first to ensure pool.json is present
  try {
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 python/generate_pool.py --n_qubits " + std::to_string(physics.get_num_qubits()) + " --n_electrons " + std::to_string(physics.get_n_electrons());
    spdlog::info("[ADAPT_sim] Generating pool: {}", command);
    int ret = system(command.c_str());
    if (ret != 0) {
      spdlog::warn("[ADAPT_sim] generate_pool.py returned non-zero code");
    }
  } catch (const std::exception& e) {
    spdlog::error("[ADAPT_sim] Failed to generate pool: {}", e.what());
  }

  auto pool = OperatorPool::load_from_json("pool.json");
  if (pool.empty()) {
    spdlog::error("[ADAPT_sim] Pool is empty, aborting.");
    return;
  }

  std::vector<double> current_params;
  int macro_iter = 0;

  while (macro_iter < max_macro_iter) {
    macro_iter++;
    spdlog::info("[ADAPT_sim] --- Macro Iteration {}/{} ---", macro_iter, max_macro_iter);
    spdlog::info("[ADAPT_sim] Evaluating gradients for {} operators in pool...", pool.size());

    std::vector<double> gradients(pool.size(), 0.0);
    double max_grad_abs = 0.0;
    int best_op_idx = -1;
    double grad_l2_norm = 0.0;

    // Gradient Evaluation via PSR
    for (size_t i = 0; i < pool.size(); ++i) {
      ansatz.add_operator(pool[i]);
      
      std::vector<double> params_plus = current_params;
      params_plus.push_back(3.14159265358979323846 / 2.0);
      
      std::vector<double> params_minus = current_params;
      params_minus.push_back(-3.14159265358979323846 / 2.0);

      std::vector<qcomp> dummy_rdm;
      double e_plus = Simulation::evaluate_functional(params_plus, &ctx, ctx.qubits, dummy_rdm);
      double e_minus = Simulation::evaluate_functional(params_minus, &ctx, ctx.qubits, dummy_rdm);

      double grad = 0.5 * (e_plus - e_minus);
      gradients[i] = grad;
      grad_l2_norm += grad * grad;

      if (std::abs(grad) > max_grad_abs) {
        max_grad_abs = std::abs(grad);
        best_op_idx = i;
      }

      ansatz.remove_last_operator();
      if (i % 10 == 0) {
        spdlog::info("[ADAPT_sim] Gradient {} out of {}", i, pool.size());
      }
    }

    grad_l2_norm = std::sqrt(grad_l2_norm);
    spdlog::info("[ADAPT_sim] Gradient L2 Norm: {:.6f} (Max abs grad: {:.6f})", grad_l2_norm, max_grad_abs);

    if (grad_l2_norm < adapt_epsilon) {
      spdlog::info("[ADAPT_sim] Convergence reached! L2 Norm ({:.6f}) < epsilon ({:.6f})", grad_l2_norm, adapt_epsilon);
      break;
    }

    // Add best operator
    spdlog::info("[ADAPT_sim] Selected Operator {} with gradient {}", best_op_idx, gradients[best_op_idx]);
    ansatz.add_operator(pool[best_op_idx]);
    current_params.push_back(0.0); // Initialize new parameter to 0

    // Macro-iteration Optimization
    nlopt::algorithm algo = nlopt::LN_NELDERMEAD;
    if (optimizer_name == "LN_PRAXIS") algo = nlopt::LN_PRAXIS;
    else if (optimizer_name == "LN_SBPLX") algo = nlopt::LN_SBPLX;
    else if (optimizer_name == "GN_DIRECT") algo = nlopt::GN_DIRECT;
    else if (optimizer_name == "GN_DIRECT_L") algo = nlopt::GN_DIRECT_L;
    else if (optimizer_name == "GN_CRS2_LM") algo = nlopt::GN_CRS2_LM;
    else if (optimizer_name == "GN_ISRES") algo = nlopt::GN_ISRES;
    else if (optimizer_name == "GN_ESCH") algo = nlopt::GN_ESCH;
    else if (optimizer_name == "LD_LBFGS" || optimizer_name == "L-BFGS") algo = nlopt::LD_LBFGS;
    else if (optimizer_name == "LD_SLSQP" || optimizer_name == "SLSQP") algo = nlopt::LD_SLSQP;

    Simulation sim(ctx, algo);
    sim.set_max_evals(max_evals);
    if (optimizer_name == "NLOPT" || optimizer_name != "SPSA") {
      sim.set_optimizer_type(Simulation::OptType::NLOPT);
      sim.set_tolerance(vqe_tol);
    } else {
      sim.set_optimizer_type(Simulation::OptType::SPSA);
    }

    // The callback provided by user is meant for this simulation instance
    // Note: If you want to use it per macro-iteration properly, you need the right function signature.
    // However, Simulation::run signature changed recently. Let's rely on the latest version.
    
    // Actually, in your modified code, Simulation::run only takes (optimal_params, callback).
    // The context is passed during Simulation constructor or directly.
    sim.run(current_params, callback);

    // After Simulation::run, current_params holds the optimized parameters.
    spdlog::info("[ADAPT_sim] Macro Iteration {} complete. Current ansatz size: {}", macro_iter, ansatz.get_num_params());

    // Write output
    std::ofstream out_file(output_json_path);
    if (out_file.is_open()) {
      json j;
      j["adapt_macro_iter"] = macro_iter;
      j["gradient_l2_norm"] = grad_l2_norm;
      j["n_params"] = ansatz.get_num_params();
      j["parameters"] = current_params;
      out_file << j.dump(4);
    }
  }

  spdlog::info("[ADAPT_sim] ADAPT-VQE Finished after {} macro iterations.", macro_iter - 1);
}
