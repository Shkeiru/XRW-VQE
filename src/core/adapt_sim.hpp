//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file adapt_sim.hpp
 * @author Rayan MALEK
 * @date 2026-05-14
 * @brief Implementation of the ADAPT-VQE orchestration loop.
 */

#pragma once

#include "ansatz.hpp"
#include "vqe_context.hpp"
#include "simulation.hpp"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

//------------------------------------------------------------------------------
//     OPERATOR POOL
//------------------------------------------------------------------------------

/**
 * @class OperatorPool
 * @brief Manages the operator pool for ADAPT-VQE.
 */
class OperatorPool {
public:
  /**
   * @brief Loads an operator pool from a JSON file.
   * @param filepath Path to the pool.json file.
   * @return A vector of operators, where each operator is a vector of GadgetInst.
   */
  static std::vector<std::vector<GadgetInst>> load_from_json(const std::string& filepath);
};

//------------------------------------------------------------------------------
//     ADAPT SIMULATOR
//------------------------------------------------------------------------------

/**
 * @class ADAPT_sim
 * @brief Orchestrates the ADAPT-VQE algorithm.
 */
class ADAPT_sim {
private:
  Physics& physics;
  std::string optimizer_name;
  int max_evals;
  double vqe_tol;
  double adapt_epsilon;
  int n_shots;
  double lambda_val;
  std::string fcalc_path;
  std::string ft_int_path;
  std::string output_json_path;
  int max_macro_iter;
  
  std::function<void(int, double, double, double, const std::vector<double>&, const std::vector<double>&)> callback;

public:
  ADAPT_sim(Physics& physics,
            std::string optimizer_name, int max_evals, double vqe_tol, double adapt_epsilon,
            int n_shots, double lambda_val,
            std::string fcalc_path, std::string ft_int_path, std::string output_json_path,
            int max_macro_iter,
            std::function<void(int, double, double, double, const std::vector<double>&, const std::vector<double>&)> callback);

  /**
   * @brief Runs the ADAPT-VQE algorithm.
   */
  void run_adapt();
};
