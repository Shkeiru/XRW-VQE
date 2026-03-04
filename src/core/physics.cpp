//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file physics.cpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Implementation of the Physics class for loading Hamiltonians.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "physics.hpp"
#include "compat.h"
#include <cstdio>
#include <iostream>
#include <sstream>


//------------------------------------------------------------------------------
//     CONSTRUCTOR
//------------------------------------------------------------------------------

Physics::Physics(const std::string &filename) : filename(filename) {
  load_hamiltonian();
}

//------------------------------------------------------------------------------
//     GETTERS
//------------------------------------------------------------------------------

int Physics::get_num_qubits() const { return num_qubits; }

int Physics::get_num_terms() const { return (int)pauli_strings.size(); }

int Physics::get_n_electrons() const { return n_electrons; }

PauliStrSum Physics::get_quest_hamiltonian() const { return quest_hamiltonian; }

const std::vector<std::string> &Physics::get_pauli_strings() const {
  return pauli_strings;
}
const std::vector<qcomp> &Physics::get_coefficients() const {
  return coefficients;
}

//------------------------------------------------------------------------------
//     HELPERS
//------------------------------------------------------------------------------

qcomp Physics::parse_coefficient(const std::string &coeff_str) {
  double real = 0.0, imag = 0.0;
  if (coeff_str.find("(") != std::string::npos) {
    sscanf(coeff_str.c_str(), "(%lf%lfj)", &real, &imag);
  } else {
    try {
      real = std::stod(coeff_str);
    } catch (...) {
      real = 0.0;
    }
  }
  return {real, imag};
}

//------------------------------------------------------------------------------
//     LOADER
//------------------------------------------------------------------------------

void Physics::load_hamiltonian() {
  spdlog::trace("Entering Physics::load_hamiltonian() to parse: {}", filename);
  spdlog::info(">>> Loading Hamiltonian from {} <<<", filename);

  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Could not open file: {}", filename);
    return;
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception &e) {
    spdlog::error("JSON parsing error: {}", e.what());
    return;
  }

  if (j.contains("n_qubits")) {
    num_qubits = j["n_qubits"].get<int>();
  } else {
    spdlog::warn("n_qubits not found in JSON, defaulting to 0");
    num_qubits = 0;
  }

  if (j.contains("n_electrons")) {
    n_electrons = j["n_electrons"].get<int>();
  } else {
    // Default to num_qubits / 2 if not found (assuming half-filling neutral) or
    // 0
    spdlog::warn("n_electrons not found in JSON, defaulting to 0");
    n_electrons = 0;
  }

  pauli_strings.clear();
  coefficients.clear();

  for (auto &[key, term] : j.items()) {
    if (key == "n_qubits" || key == "n_orbitals" || key == "multiplicity" ||
        key == "charge" || key == "basis" || key == "status" || key == "file" ||
        key == "n_terms")
      continue;

    if (!term.contains("pauli_string") || !term.contains("coefficient"))
      continue;

    std::string pauli = term["pauli_string"].get<std::string>();
    std::string coeff_str = term["coefficient"].get<std::string>();

    pauli_strings.push_back(pauli);
    coefficients.push_back(parse_coefficient(coeff_str));
  }

  // Create QuEST Hamiltonian
  std::vector<PauliStr> terms;
  for (const auto &s : pauli_strings) {
    if (s == "I") {
      // Identity: QuEST requires at least one operator.
      // Specifying I on qubit 0 is equivalent to global Identity (since others
      // result in I by default). We assume num_qubits >= 1.
      std::vector<int> indices = {0};
      terms.push_back(getPauliStr("I", indices));
    } else {
      std::string codes;
      std::vector<int> indices;

      std::stringstream ss(s);
      std::string token;
      while (ss >> token) {
        if (token.length() < 2)
          continue;
        char op = token[0];
        try {
          int idx = std::stoi(token.substr(1));
          codes += op;
          indices.push_back(idx);
        } catch (...) {
          spdlog::warn("Failed to parse token: {}", token);
        }
      }

      if (codes.empty()) {
        // Fallback if parsing failed or string was empty but not "I"
        // Treat as Identity to avoid crash
        std::vector<int> idx = {0};
        terms.push_back(getPauliStr("I", idx));
      } else {
        terms.push_back(getPauliStr(codes, indices));
      }
    }
  }

  if (!terms.empty()) {
    quest_hamiltonian =
        createPauliStrSum(terms.data(), coefficients.data(), terms.size());
  } else {
    // Handle empty if necessary
  }

  spdlog::info("Hamiltonian loaded: {} qubits, {} terms", num_qubits,
               terms.size());
  spdlog::trace("Exiting Physics::load_hamiltonian() successfully");
}
