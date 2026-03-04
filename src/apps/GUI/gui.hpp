#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "ImGuiFileDialog.h"
#include "compat.h"
#include "imgui.h"
#include "implot.h"
#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <nlopt.hpp>
#include <quest.h>
#include <string>
#include <thread>
#include <vector>

//------------------------------------------------------------------------------
//     CLASS DECLARATION
//------------------------------------------------------------------------------

/**
 * @class GUI
 * @brief Manages the graphical user interface for the VQE Simulator.
 *
 * This class handles the rendering of the GUI, user interactions, configuration
 * settings, and visualization of simulation results. It integrates with ImGui
 * for the interface and interacts with the simulation core.
 *
 * @author Rayan MALEK
 * @date 2026-02-19
 */
class GUI {
public:
  /**
   * @brief Constructs a new GUI object.
   *
   * Initializes the simulation environment and sets up default values.
   */
  GUI();

  /**
   * @brief Destroy the GUI object.
   *
   * Cleans up resources, including joining the calculation thread and
   * finalizing the QuEST environment.
   */
  ~GUI();

  /**
   * @brief Main render function.
   *
   * This function should be called inside the main application loop to render
   * the GUI elements each frame.
   */
  void Render();

private:
  //----------------------------------------------------------------------------
  //     LAYOUT METHODS
  //----------------------------------------------------------------------------

  /**
   * @brief Draws the configuration window.
   *
   * Handles user inputs for molecule parameters, VQE settings, and ansatz
   * selection.
   */
  void DrawConfiguration();

  /**
   * @brief Draws the information and results window.
   *
   * Displays the logging region, current simulation status, and numerical
   * results.
   */
  void DrawInfoAndResults();

  /**
   * @brief Draws the plots window.
   *
   * Renders the energy convergence graph and the probability histogram.
   */
  void DrawPlots();

  //----------------------------------------------------------------------------
  //     HELPER METHODS
  //----------------------------------------------------------------------------

  /**
   * @brief Logs a message to the internal log buffer.
   *
   * @param message The message string to log.
   */
  void Log(const std::string &message);

  //----------------------------------------------------------------------------
  //     ACTION METHODS
  //----------------------------------------------------------------------------

  /**
   * @brief Saves the current simulation run to a JSON file.
   *
   * Exports configuration, results, history, and state probabilities.
   */
  void SaveRun();

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - CONFIGURATION
  //----------------------------------------------------------------------------

  char atom_string[512] =
      "H 0 0 0; H 0 0 0.735"; ///< PySCF atom geometry string.
  char basis[128] = "sto-3g"; ///< Basis set name.
  int charge = 0;             ///< Molecular charge.
  int spin = 0;               ///< Molecular spin (2S).
  int mapping_idx = 0;        ///< Selected mapping index.
  std::vector<const char *> mappings = {"Jordan-Wigner", "Bravyi-Kitaev"};

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - DIFFRACTION
  //----------------------------------------------------------------------------

  std::string filepath_integrals = ""; ///< Path to integrals file
  std::string filepath_factors = "";   ///< Path to experimental factors file
  bool is_diffraction_ready = false;   ///< Flag if both files are selected
  double lambda_diffraction = 1.0; ///< Scaling factor for diffraction penalty

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - NLOPT
  //----------------------------------------------------------------------------

  std::vector<const char *> optimizers = {
      "LN_COBYLA",   "LN_BOBYQA",     "LN_NEWUOA", "LN_NEWUOA_BOUND",
      "LN_PRAXIS",   "LN_NELDERMEAD", "LN_SBPLX",  "GN_DIRECT",
      "GN_DIRECT_L", "GN_CRS2_LM",    "GN_ISRES",  "GN_ESCH",
      "LD_LBFGS",    "LD_SLSQP"};

  std::vector<nlopt::algorithm> optimizer_enums = {
      nlopt::LN_COBYLA,       nlopt::LN_BOBYQA, nlopt::LN_NEWUOA,
      nlopt::LN_NEWUOA_BOUND, nlopt::LN_PRAXIS, nlopt::LN_NELDERMEAD,
      nlopt::LN_SBPLX,        nlopt::GN_DIRECT, nlopt::GN_DIRECT_L,
      nlopt::GN_CRS2_LM,      nlopt::GN_ISRES,  nlopt::GN_ESCH,
      nlopt::LD_LBFGS,        nlopt::LD_SLSQP};

  int optimizer_idx = 5;   ///< Default optimizer index (Nelder-Mead).
  int max_iter = 100;      ///< Maximum number of iterations for the optimizer.
  int shots = 1024;        ///< Number of shots for quantum measurement.
  double tolerance = 1e-8; ///< Tolerance for convergence.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - INFO
  //----------------------------------------------------------------------------

  int num_qubits = 0;          ///< Number of qubits in the system.
  int num_paulis = 0;          ///< Number of Pauli terms in the Hamiltonian.
  long long hilbert_space = 0; ///< Size of the Hilbert space.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - RESULTS
  //----------------------------------------------------------------------------

  int current_iter = 0;        ///< Current iteration number.
  double current_energy = 0.0; ///< Current energy value.
  double best_energy = 1e9; ///< Best recorded energy (initialized to infinity).
  std::string status_message = "Prêt."; ///< Status message displayed in GUI.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - PLOTS
  //----------------------------------------------------------------------------

  std::vector<double> iter_history; ///< History of iterations for plotting.
  std::vector<double>
      energy_history; ///< History of energy values for plotting.
  std::vector<std::vector<double>>
      probs_history; ///< Probabilities at each iteration.
  std::vector<std::vector<double>>
      params_history; ///< Parameters at each iteration.

  std::vector<const char *> counts_labels; ///< Labels for histogram bars.
  std::vector<double> counts_values; ///< Probability values for histogram.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - UI STATE
  //----------------------------------------------------------------------------

  ImVec4 graph_bg_color =
      ImVec4(0.1f, 0.1f, 0.1f, 1.0f); ///< Background color for plots.
  int hea_depth = 3;                  ///< Depth of the HEA ansatz.
  bool log_scale_P =
      false; ///< Flag for logarithmic scale in probability plots.
  bool log_scale_E = false; ///< Flag for logarithmic scale in energy plots.
  int ansatz_idx = 0;       ///< Selected ansatz index.
  std::vector<const char *> ansatz_types = {"HEA", "UCCSD"};

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - CONCURRENCY
  //----------------------------------------------------------------------------

  std::atomic<bool> is_running =
      false;                      ///< Flag indicating if simulation is running.
  std::thread calculation_thread; ///< Worker thread for simulation.
  std::mutex graph_mutex;         ///< Mutex for thread-safe data access.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - QUEST
  //----------------------------------------------------------------------------

  QuESTEnv env; ///< QuEST environment.

  //----------------------------------------------------------------------------
  //     DATA MEMBERS - SIMULATION RESULTS
  //----------------------------------------------------------------------------

  /**
   * @struct NoiseResult
   * @brief Stores the results of the simulation, including noise statistics.
   */
  struct NoiseResult {
    double variance = 0;     ///< Variance of the energy.
    double noise_std = 0;    ///< Standard deviation of the noise.
    double noisy_energy = 0; ///< Energy calculated with noise.
    std::vector<double>
        sampled_probs; ///< Probabilities sampled from the final state.
  } final_results;

  bool hamiltonian_exists =
      false; ///< Flag indicating if Hamiltonian has been generated.
};
