//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file gui.cpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Implementation of the GUI class for the VQE Simulator.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "gui.hpp"
#include "../core/logger.hpp"
#include "../core/simulation.hpp"
#include "../core/adapt_sim.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

//------------------------------------------------------------------------------
//     CONSTRUCTOR / DESTRUCTOR
//------------------------------------------------------------------------------

GUI::GUI() {
  initQuESTEnv();
  env = getQuESTEnv();
}

GUI::~GUI() {
  if (calculation_thread.joinable())
    calculation_thread.join();
  finalizeQuESTEnv();
}

//------------------------------------------------------------------------------
//     RENDER LOOP
//------------------------------------------------------------------------------

void GUI::Render() {
  // Determine viewport size
  ImVec2 viewport_size = ImGui::GetMainViewport()->Size;

  // 1. Configuration Window
  ImGui::Begin("Configuration");
  DrawConfiguration();
  ImGui::End();

  // 2. Info & Results Window
  ImGui::Begin("Info & Results");
  DrawInfoAndResults();
  ImGui::End();

  // 3. Plots Window
  ImGui::Begin("Plots");
  DrawPlots();
  ImGui::End();
}

//------------------------------------------------------------------------------
//     CONFIGURATION WINDOW
//------------------------------------------------------------------------------

void GUI::DrawConfiguration() {
  ImGui::BeginChild("Col1_Config");

  //--------------------------------------------------------------------------
  // Molecule Configuration
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("1. Molecule");
  ImGui::Text("Atom String (PySCF):");
  ImGui::InputText("##atom", atom_string, sizeof(atom_string));

  ImGui::Text("Basis Set:");
  ImGui::InputText("##basis", basis, sizeof(basis));

  ImGui::Text("Charge / Spin (2S):");
  ImGui::InputInt("Charge", &charge);
  ImGui::InputInt("Spin", &spin);

  ImGui::Text("Mapping Qubit:");
  ImGui::Combo("##map", &mapping_idx, mappings.data(), mappings.size());

  // Generation Hamiltonien
  if (ImGui::Button("Generer Hamiltonien")) {
    spdlog::info(">>> Generation Hamiltonien...");
    status_message = "Generation en cours...";

    std::string cmd_atom(atom_string);
    std::string cmd_basis(basis);
    std::string cmd_map = mappings[mapping_idx];

    // Format mapping string: lowercase and replace '-' with '_'
    std::transform(cmd_map.begin(), cmd_map.end(), cmd_map.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::replace(cmd_map.begin(), cmd_map.end(), '-', '_');

    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 python/generate_hamiltonian.py";
    command += " --atom \"" + cmd_atom + "\"";
    command += " --basis " + cmd_basis;
    command += " --charge " + std::to_string(charge);
    command += " --spin " + std::to_string(spin);
    command += " --mapping " + cmd_map;

    spdlog::info("CMD: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (!pipe) {
      spdlog::error(
          "Impossible d'ouvrir le pipe pour la generation d'Hamiltonien.");
      status_message = "Erreur Pipe.";
    } else {
      char buffer[256];
      while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        std::string msg(buffer);
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
          msg.pop_back();
        }
        if (!msg.empty()) {
          spdlog::info("PySCF: {}", msg);
        }
      }
      int ret = _pclose(pipe);
      if (ret != 0) {
        spdlog::error("Erreur lors de l'execution du script Python. Code: {}",
                      ret);
        status_message = "Erreur Generation.";
      } else {
        // Parse JSON output
        std::ifstream f("hamiltonian.json");
        if (f.good()) {
          try {
            nlohmann::json j;
            f >> j;
            if (j.contains("error")) {
              spdlog::error("Erreur script Python: {}",
                            j["error"].get<std::string>());
              status_message = "Erreur Script.";
            } else {
              num_paulis = 0;
              for (auto &[key, val] : j.items()) {
                if (key.find("term_") == 0)
                  num_paulis++;
              }
              if (j.contains("n_qubits")) {
                num_qubits = j["n_qubits"].get<int>();
              }
              hilbert_space = (long long)std::pow(2, num_qubits);
              status_message = "Hamiltonien Genere.";
              hamiltonian_exists = true;
              spdlog::info("Hamiltonien charge. Qubits: {}, Termes: {}",
                           num_qubits, num_paulis);
            }
          } catch (const std::exception &e) {
            spdlog::error("Erreur JSON: {}", e.what());
            status_message = "Erreur Parsing.";
          }
        } else {
          spdlog::error("Fichier hamiltonian.json introuvable.");
          status_message = "Fichier manquant.";
        }
      }
    }
  }

  ImGui::TextWrapped("%s", status_message.c_str());

  ImGui::Spacing();

  //--------------------------------------------------------------------------
  // VQE Parameters
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("2. Parametres VQE");

  ImGui::Text("Optimiseur:");
  ImGui::Combo("##opt", &optimizer_idx, optimizers.data(), optimizers.size());

  if (std::string(optimizers[optimizer_idx]) == "SPSA") {
    ImGui::SeparatorText("SPSA Hyperparametres");
    ImGui::InputDouble("Step size (a)", &spsa_a, 0.0, 0.0, "%.3f");
    ImGui::InputDouble("Perturbation (c)", &spsa_c, 0.0, 0.0, "%.3f");
    ImGui::InputDouble("Stability (A)", &spsa_A, 0.0, 0.0, "%.3f");
    ImGui::InputDouble("Decay alpha", &spsa_alpha, 0.0, 0.0, "%.3f");
    ImGui::InputDouble("Decay gamma", &spsa_gamma, 0.0, 0.0, "%.3f");
    ImGui::Spacing();
  }

  ImGui::Text("Iterations Optimiseur:");
  ImGui::InputInt("##iter", &max_iter);

  ImGui::Text("Tolerance:");
  ImGui::InputDouble("##tol", &tolerance, 0.0, 0.0, "%.1e");

  ImGui::Text("Nombre de Shots:");
  ImGui::InputInt("##shots", &shots);
  if (shots < 0)
    shots = 0;

  ImGui::Spacing();
  ImGui::SeparatorText("2.5 ADAPT-VQE");
  ImGui::Checkbox("Enable ADAPT-VQE", &use_adapt);
  if (use_adapt) {
    ImGui::Text("ADAPT Gradient L2 Tolerance:");
    ImGui::InputDouble("##adapt_tol", &adapt_epsilon, 0.0, 0.0, "%.1e");
  }

  ImGui::Spacing();

  //--------------------------------------------------------------------------
  // Ansatz Configuration
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("3. Ansatz");
  ImGui::Combo("Type", &ansatz_idx, ansatz_types.data(), ansatz_types.size());

  if (ansatz_idx == 0) { // HEA
    ImGui::SliderInt("Depth", &hea_depth, 1, 20);
  } else { // UCCSD
    ImGui::Text("UCCSD utilise les excitations simples et doubles.");
    ImGui::Text("Mapping: %s", mappings[mapping_idx]);
  }

  ImGui::Spacing();
  if (hamiltonian_exists) {
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Hamiltonien OK");
  } else {
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Hamiltonien manquant");
  }

  //--------------------------------------------------------------------------
  // Warm Start (Optional)
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("3.5 Warm Start (Optionnel)");
  ImGui::Checkbox("Utiliser Warm Start", &use_warm_start);
  if (use_warm_start) {
    if (ImGui::Button("Choisir Fichier Warm Start")) {
      IGFD::FileDialogConfig config;
      config.path = ".";
      ImGuiFileDialog::Instance()->OpenDialog(
          "ChooseWarmStart", "Choisir Fichier Warm Start", ".json", config);
    }
    ImGui::SameLine();
    ImGui::TextWrapped(
        "%s",
        filepath_warm_start.empty() ? "Aucun" : filepath_warm_start.c_str());
  }

  //--------------------------------------------------------------------------
  // Diffraction Data (File Selection)
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("1.5 Donnees de Diffraction");

  // Selection Integrales
  if (ImGui::Button("Choisir Fichier Integrales")) {
    IGFD::FileDialogConfig config;
    config.path = ".";
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseIntegrals", "Choisir Fichier Integrales", ".*", config);
  }
  ImGui::SameLine();
  ImGui::TextWrapped(
      "%s", filepath_integrals.empty() ? "Aucun" : filepath_integrals.c_str());

  // Selection Facteurs experimentaux
  if (ImGui::Button("Choisir Fichier Facteurs")) {
    IGFD::FileDialogConfig config;
    config.path = ".";
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseFactors", "Choisir Fichier Facteurs Expérimentaux", ".*",
        config);
  }
  ImGui::SameLine();
  ImGui::TextWrapped("%s", filepath_factors.empty() ? "Aucun"
                                                    : filepath_factors.c_str());

  ImGui::Spacing();
  ImGui::Text("Facteur Lambda (Diffraction):");
  ImGui::InputDouble("##lambda", &lambda_diffraction, 0.0, 0.0, "%.2f");

  // Maj du statut global de diffraction
  is_diffraction_ready =
      (!filepath_integrals.empty() && !filepath_factors.empty());

  if (is_diffraction_ready) {
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Donnees de diffraction pretes !");
  }

  // Draw dialogs
  if (ImGuiFileDialog::Instance()->Display("ChooseWarmStart")) {
    if (ImGuiFileDialog::Instance()->IsOk()) {
      filepath_warm_start = ImGuiFileDialog::Instance()->GetFilePathName();
    }
    ImGuiFileDialog::Instance()->Close();
  }

  if (ImGuiFileDialog::Instance()->Display("ChooseIntegrals")) {
    if (ImGuiFileDialog::Instance()->IsOk()) {
      filepath_integrals = ImGuiFileDialog::Instance()->GetFilePathName();
    }
    ImGuiFileDialog::Instance()->Close();
  }

  if (ImGuiFileDialog::Instance()->Display("ChooseFactors")) {
    if (ImGuiFileDialog::Instance()->IsOk()) {
      filepath_factors = ImGuiFileDialog::Instance()->GetFilePathName();
    }
    ImGuiFileDialog::Instance()->Close();
  }

  ImGui::Spacing();

  //--------------------------------------------------------------------------
  // Run Simulation
  //--------------------------------------------------------------------------
  if (ImGui::Button("RUN VQE", ImVec2(-FLT_MIN, 0))) {
    if (!hamiltonian_exists) {
      spdlog::error("Veuillez d'abord generer l'Hamiltonien.");
    } else if (is_running) {
      spdlog::warn("Simulation deja en cours...");
    } else {
      spdlog::info(">>> Debut VQE... (Simulation)");
      status_message = "VQE Running...";

      // Reset Data
      iter_history.clear();
      energy_history.clear();
      base_energy_history.clear();
      chi_squared_history.clear();
      probs_history.clear();
      params_history.clear();
      current_iter = 0;
      current_energy = 0.0;
      current_base_energy = 0.0;
      current_chi_squared = 0.0;
      best_energy = 1e9;
      counts_values.clear();
      macro_boundaries.clear();
      macro_colors.clear();

      // Pre-reserve history vectors to avoid repeated allocations
      iter_history.reserve(max_iter);
      energy_history.reserve(max_iter);
      base_energy_history.reserve(max_iter);
      chi_squared_history.reserve(max_iter);
      probs_history.reserve(max_iter);
      params_history.reserve(max_iter);

      // Launch Thread
      is_running = true;
      if (calculation_thread.joinable())
        calculation_thread.join();

      nlopt::algorithm selected_algo = optimizer_enums[optimizer_idx];
      std::string current_optimizer_name = optimizers[optimizer_idx];
      bool is_spsa = (current_optimizer_name == "SPSA");
      double s_a = spsa_a, s_c = spsa_c, s_A = spsa_A, s_alpha = spsa_alpha, s_gamma = spsa_gamma;

      // Capture necessary values by value for thread safety
      int current_ansatz_idx = ansatz_idx;
      int current_depth = hea_depth;
      std::string current_map = mappings[mapping_idx];
      std::string integrals_path =
          is_diffraction_ready ? filepath_integrals : "";
      std::string factors_path = is_diffraction_ready ? filepath_factors : "";
      double current_lambda = lambda_diffraction;

      calculation_thread = std::thread(
          [this, selected_algo, current_optimizer_name, current_ansatz_idx, current_depth, current_map,
           integrals_path, factors_path, current_lambda, is_spsa, s_a, s_c, s_A,
           s_alpha, s_gamma,
           ws_enabled = use_warm_start,
           ws_path = use_warm_start ? filepath_warm_start : std::string("")]() {
            try {
              // 1. Load Physics
              Physics physics("hamiltonian.json");

              // 2. Setup Ansatz
              std::unique_ptr<Ansatz> ansatz;
              if (current_ansatz_idx == 0) {
                ansatz = std::make_unique<HEA>(physics.get_num_qubits(),
                                               current_depth);
              } else {
                // Format mapping string
                std::string map_str = current_map;
                std::transform(map_str.begin(), map_str.end(), map_str.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                std::replace(map_str.begin(), map_str.end(), '-', '_');

                ansatz =
                    std::make_unique<UCCSD>(physics.get_num_qubits(),
                                            physics.get_n_electrons(), map_str);
              }

              // 4. Create VQEContext
              // If ADAPT, we defer to ADAPT_sim.
              std::unique_ptr<VQEContext> ctx;
              std::unique_ptr<Simulation> sim;

              if (use_adapt) {
                spdlog::info(">>> Starting ADAPT-VQE Mode...");
                ADAPT_sim adapt_sim(physics,
                                    current_optimizer_name, max_iter, tolerance, adapt_epsilon,
                                    shots, current_lambda, factors_path, integrals_path,
                                    "adapt_gui_out.json", 10,
                                    [this](int iter, double total_energy, double quantum_energy,
                                           double chi_squared, const std::vector<double> &probs,
                                           const std::vector<double> &cb_params) {
                                      std::lock_guard<std::mutex> lock(graph_mutex);
                                      if (iter == 1 || iter_history.empty()) {
                                        macro_boundaries.push_back(iter_history.size());
                                        // Generate random color for this macro iteration
                                        float r = 0.3f + (float)rand() / (float)RAND_MAX * 0.7f;
                                        float g = 0.3f + (float)rand() / (float)RAND_MAX * 0.7f;
                                        float b = 0.3f + (float)rand() / (float)RAND_MAX * 0.7f;
                                        macro_colors.push_back(ImVec4(r, g, b, 1.0f));
                                      }
                                      iter_history.push_back((double)(iter_history.size() + 1));
                                      energy_history.push_back(total_energy);
                                      base_energy_history.push_back(quantum_energy);
                                      chi_squared_history.push_back(chi_squared);
                                      probs_history.push_back(probs);
                                      params_history.push_back(cb_params);
                                      current_energy = total_energy;
                                      current_base_energy = quantum_energy;
                                      current_chi_squared = chi_squared;
                                      counts_values = probs;
                                      current_iter = iter;
                                      if (total_energy < best_energy)
                                        best_energy = total_energy;
                                    });
                adapt_sim.run_adapt();

                std::lock_guard<std::mutex> lock(graph_mutex);
                is_running = false;
                status_message = "ADAPT-VQE Termine.";
              } else {
                ctx = std::make_unique<VQEContext>(physics, *ansatz);
                ctx->n_shots = shots;
                ctx->lambda = current_lambda;
                ctx->setup(factors_path, integrals_path);

                sim = std::make_unique<Simulation>(*ctx, selected_algo);
                if (is_spsa) {
                  sim->set_optimizer_type(Simulation::OptType::SPSA);
                  SPSAParams params_spsa;
                  params_spsa.a = s_a;
                  params_spsa.c = s_c;
                  params_spsa.A = s_A;
                  params_spsa.alpha = s_alpha;
                  params_spsa.gamma = s_gamma;
                  sim->set_spsa_params(params_spsa);
                } else {
                  sim->set_optimizer_type(Simulation::OptType::NLOPT);
                }
                sim->set_max_evals(max_iter);
                sim->set_tolerance(tolerance);

                // 4. Run Optimization
                std::vector<double> params(ansatz->get_num_params(), 0);

                // Small random perturbation to break zero-initialization symmetry
                {
                  std::mt19937 rng(std::random_device{}());
                  std::uniform_real_distribution<double> dist(-1e-3, 1e-3);
                  for (auto& p : params) p = dist(rng);
                  spdlog::info("Initial parameters perturbed with uniform noise in [-0.001, 0.001] rad");
                }

                // --- Warm Start: Load parameters if enabled ---
                if (ws_enabled && !ws_path.empty()) {
                  try {
                    std::ifstream ws_file(ws_path);
                    if (!ws_file.good()) {
                      spdlog::error("Warm start file not found: {}", ws_path);
                    } else {
                      nlohmann::json ws_json;
                      ws_file >> ws_json;

                      auto tag = ws_json["ansatz_tag"];
                      std::string ws_type = tag["type"].get<std::string>();
                      int ws_nq = tag["num_qubits"].get<int>();
                      std::string expected_type =
                          (current_ansatz_idx == 0) ? "HEA" : "UCCSD";

                      bool valid = true;
                      if (ws_type != expected_type) {
                        spdlog::error(
                            "Warm start ansatz mismatch: file={}, current={}",
                            ws_type, expected_type);
                        valid = false;
                      }
                      if (ws_nq != (int)physics.get_num_qubits()) {
                        spdlog::error(
                            "Warm start num_qubits mismatch: file={}, current={}",
                            ws_nq, physics.get_num_qubits());
                        valid = false;
                      }
                      if (valid && ws_type == "HEA" &&
                          tag["depth"].get<int>() != current_depth) {
                        spdlog::error(
                            "Warm start HEA depth mismatch: file={}, current={}",
                            tag["depth"].get<int>(), current_depth);
                        valid = false;
                      }
                      if (valid && ws_type == "UCCSD" &&
                          tag["num_electrons"].get<int>() !=
                              (int)physics.get_n_electrons()) {
                        spdlog::error(
                            "Warm start UCCSD num_electrons mismatch: file={}, "
                            "current={}",
                            tag["num_electrons"].get<int>(),
                            physics.get_n_electrons());
                        valid = false;
                      }

                      if (valid) {
                        auto ws_params =
                            ws_json["parameters"].get<std::vector<double>>();
                        if ((int)ws_params.size() != ansatz->get_num_params()) {
                          spdlog::error(
                              "Warm start params size mismatch: file={}, "
                              "expected={}",
                              ws_params.size(), ansatz->get_num_params());
                        } else {
                          params = ws_params;
                          spdlog::info(
                              "Warm start loaded: {} parameters from {}",
                              params.size(), ws_path);
                        }
                      }
                    }
                  } catch (const std::exception &e) {
                    spdlog::error("Warm start loading failed: {}", e.what());
                  }
                }

                double min_energy = sim->run(
                    params,
                    [this](int iter, double total_energy, double quantum_energy,
                           double chi_squared, const std::vector<double> &probs,
                           const std::vector<double> &cb_params) {
                      std::lock_guard<std::mutex> lock(graph_mutex);
                      iter_history.push_back((double)iter);
                      energy_history.push_back(total_energy);
                      base_energy_history.push_back(quantum_energy);
                      chi_squared_history.push_back(chi_squared);
                      probs_history.push_back(probs);
                      params_history.push_back(cb_params);
                      current_energy = total_energy;
                      current_base_energy = quantum_energy;
                      current_chi_squared = chi_squared;
                      counts_values = probs;
                      current_iter = iter;
                      if (total_energy < best_energy)
                        best_energy = total_energy;
                    });

                // 5. Update Status with Noise Results (if any)
                try {
                  std::lock_guard<std::mutex> lock(graph_mutex);
                  is_running = false;
                  status_message = "VQE Termine.";

                  // If shots > 0, min_energy IS the noisy energy from the last
                  // step
                  final_results.noisy_energy = min_energy;
                  final_results.variance = sim->get_last_variance();
                  final_results.noise_std = sim->get_last_std();

                  // Get probabilities for final params
                  final_results.sampled_probs = sim->get_probabilities(params);
                  
                  // Get evaluated RDMs
                  final_results.rdms = sim->get_rdms();
                } catch (const std::exception &e) {
                  std::lock_guard<std::mutex> lock(graph_mutex);
                  is_running = false;
                  status_message = "Erreur VQE.";
                  spdlog::error("Erreur VQE Probabilites: {}", e.what());
                }
              }
            } catch (const std::exception &e) {
              std::lock_guard<std::mutex> lock(graph_mutex);
              is_running = false;
              status_message = "Erreur fatale VQE.";
              spdlog::error("Erreur VQE Globale: {}", e.what());
            }
          });
    }
  }

  ImGui::Spacing();
  if (ImGui::Button("Sauvegarder Run", ImVec2(-FLT_MIN, 0))) {
    SaveRun();
  }

  ImGui::Spacing();

  //--------------------------------------------------------------------------
  // Theme Selection
  //--------------------------------------------------------------------------
  ImGui::SeparatorText("4. Themes");
  ImGui::Text("Application:");
  if (ImGui::Button("Dark"))
    ImGui::StyleColorsDark();
  ImGui::SameLine();
  if (ImGui::Button("Light"))
    ImGui::StyleColorsLight();
  ImGui::SameLine();
  if (ImGui::Button("Classic"))
    ImGui::StyleColorsClassic();

  ImGui::Text("Graphe Fond:");
  ImGui::ColorEdit4("##bg", (float *)&graph_bg_color);

  ImGui::Checkbox("Echelle Logarithmique Energie", &log_scale_E);
  ImGui::Checkbox("Echelle Logarithmique Probabilités", &log_scale_P);

  ImGui::EndChild();
}

//------------------------------------------------------------------------------
//     INFO AND RESULTS WINDOW
//------------------------------------------------------------------------------

void GUI::DrawInfoAndResults() {
  ImGui::BeginChild("Col2_Info");

  ImGui::SeparatorText("Hamiltonien Info");
  if (num_qubits > 0) {
    ImGui::Text("Qubits: %d", num_qubits);
    ImGui::Text("Pauli Terms: %d", num_paulis);
    ImGui::Text("Hilbert Space: %lld", hilbert_space);
  } else {
    ImGui::TextDisabled("Non genere");
  }

  ImGui::Spacing();
  ImGui::SeparatorText("Resultats Simulation");
  if (!iter_history.empty() || status_message == "VQE Running...") {
    ImGui::Text("Iteration: %d", current_iter);
    ImGui::Text("Energie Totale: %.6f Ha", current_energy);
    ImGui::Text("Energie de Base: %.6f Ha", current_base_energy);
    ImGui::Text("Chi2: %.6e", current_chi_squared);
    ImGui::Text("Meilleure Energie: %.6f Ha", best_energy);
  } else {
    ImGui::TextDisabled("En attente de run...");
  }

  if (status_message == "VQE Termine.") {
    ImGui::SeparatorText("Resultats Bruites");
    ImGui::Text("Energie (Shot Noise): %.6f", final_results.noisy_energy);
    ImGui::Text("Variance: %.6e", final_results.variance);
    ImGui::Text("Ecart-Type: %.6e", final_results.noise_std);
  }

  ImGui::Spacing();
  ImGui::SeparatorText("Logs Application");

  ImGui::BeginChild("LogRegion", ImVec2(0, -1), true,
                    ImGuiWindowFlags_HorizontalScrollbar);

  if (qb_log::gui_sink) {
    auto recent_logs = qb_log::gui_sink->last_formatted();
    for (const auto &log_str : recent_logs) {
      ImGui::TextUnformatted(log_str.c_str());
    }
  } else {
    ImGui::TextDisabled("Logger not initialized...");
  }

  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }
  ImGui::EndChild();

  ImGui::EndChild();
}

//------------------------------------------------------------------------------
//     PLOTS WINDOW
//------------------------------------------------------------------------------

void GUI::DrawPlots() {
  ImPlot::PushStyleColor(ImPlotCol_PlotBg, graph_bg_color);

  // 1. Energy Plot (Top Half)
  float plot_height = (ImGui::GetContentRegionAvail().y / 2.0f) - 10.0f;
  if (ImPlot::BeginPlot("Convergences Energie", ImVec2(-1, plot_height))) {
    ImPlot::SetupAxes("Iterations", "Energie (Ha)", ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);

    if (log_scale_E) {
      ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    } else {
      ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
    }

    std::lock_guard<std::mutex> lock(graph_mutex);
    if (!iter_history.empty()) {
      if (macro_boundaries.empty()) {
        // Standard VQE or fallback: plot single line
        ImPlot::PlotLine(
            "Energie", iter_history.data(), energy_history.data(),
            iter_history.size());
      } else {
        // ADAPT-VQE: plot segmented lines with different colors
        for (size_t i = 0; i < macro_boundaries.size(); ++i) {
          int start_idx = macro_boundaries[i];
          int end_idx = (i + 1 < macro_boundaries.size()) ? macro_boundaries[i + 1] : iter_history.size();
          int count = end_idx - start_idx;
          if (count > 0) {
            std::string label = "Macro Iter " + std::to_string(i + 1);
            ImPlot::PlotLine(label.c_str(), &iter_history[start_idx], &energy_history[start_idx], count, ImPlotSpec(ImPlotProp_LineColor, macro_colors[i]));
          }
        }
      }
    }
    ImPlot::EndPlot();
  }

  ImGui::Spacing();

  // 2. Histogram Plot (Bottom Half)
  if (ImPlot::BeginPlot("Etat Quantique", ImVec2(-1, -1))) {
    // Setup Axes
    ImPlot::SetupAxes("Etats de Base",
                      "Probabilite"); // Remove AutoFit flags to allow zoom/pan

    // Log Scale Logic
    if (log_scale_P) {
      ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
      // In log scale, avoid 0 limits.
    } else {
      ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
      ImPlot::SetupAxisLimits(ImAxis_Y1, 1e-15, 1.0, ImPlotCond_Once);
    }

    // X-Axis Logic (Full Range)
    double max_x = (num_qubits > 0) ? std::pow(2, num_qubits) : 1.0;
    // Force set range if num_qubits changed
    static int last_num_qubits = -1;
    ImPlotCond cond =
        (num_qubits != last_num_qubits) ? ImPlotCond_Always : ImPlotCond_Once;
    last_num_qubits = num_qubits;

    ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, max_x - 0.5, cond);

    // Determine which data to show
    const std::vector<double> *current_data = &counts_values;
    if (status_message == "VQE Termine." &&
        !final_results.sampled_probs.empty()) {
      current_data = &final_results.sampled_probs;
    }

    if (!current_data->empty()) {
      // Fake x positions 0, 1, 2...
      std::vector<double> x_pos(current_data->size());
      for (size_t i = 0; i < x_pos.size(); ++i)
        x_pos[i] = (double)i;

      // Handle Zeros in Log Scale (ImPlot might clip them, or we clamp)
      std::vector<double> plot_data = *current_data;
      if (log_scale_P) {
        for (auto &v : plot_data) {
          if (v <= 0)
            v = 1e-15; // Small epsilon for log plot
        }
      }

      ImPlot::PlotBars("Proba", x_pos.data(), plot_data.data(),
                       plot_data.size(), 0.5);

      // Tooltips
      if (ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        int idx = (int)std::round(mouse.x);
        if (idx >= 0 &&
            idx < current_data->size()) { // Check against original size
          ImGui::BeginTooltip();
          // Use member num_qubits
          std::string bitstring = "";
          for (int b = 0; b < num_qubits; ++b) {
            bitstring += ((idx >> (num_qubits - 1 - b)) & 1) ? "1" : "0";
          }
          ImGui::Text("Etat: |%s>", bitstring.c_str());
          ImGui::Text("Proba: %.4e",
                      (*current_data)[idx]); // Scientific for log
          if (status_message == "VQE Termine." && shots > 0) {
            int count = (int)std::round((*current_data)[idx] * shots);
            ImGui::Text("Count: %d / %d", count, shots);
          }
          ImGui::EndTooltip();
        }
      }
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleColor();
}

//------------------------------------------------------------------------------
//     SAVE RUN
//------------------------------------------------------------------------------

void GUI::SaveRun() {
  // 1. Timestamp
  std::time_t now = std::time(nullptr);
  char buf[100];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
  std::string timestamp(buf);

  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
  std::string filename_ts(buf);

  // 2. JSON Structure
  nlohmann::json j;

  // Config
  j["config"]["molecule"] = {{"atom_string", atom_string},
                             {"basis", basis},
                             {"charge", charge},
                             {"spin", spin},
                             {"mapping", mappings[mapping_idx]}};

  j["config"]["vqe"] = {{"optimizer", optimizers[optimizer_idx]},
                        {"max_iterations", max_iter},
                        {"shots", shots},
                        {"hea_depth", hea_depth},
                        {"ansatz", "HEA"}};

  // Results (Excluding Variance/STD per user request)
  j["results"] = {
      {"final_energy", final_results.noisy_energy}, // Or best_energy? Noisy
                                                    // energy is the final
      // result of shot sim.
      {"final_base_energy",
       base_energy_history.empty() ? 0.0 : base_energy_history.back()},
      {"final_chi_squared",
       chi_squared_history.empty() ? 0.0 : chi_squared_history.back()},
      {"best_exact_energy", best_energy},
      {"status", status_message}};

  // j["rdms"] = final_results.rdms;

  // History
  std::vector<nlohmann::json> history;
  {
    std::lock_guard<std::mutex> lock(graph_mutex);
    for (size_t i = 0; i < iter_history.size(); ++i) {
      nlohmann::json entry;
      entry["iteration"] = iter_history[i];
      entry["energy"] = energy_history[i];
      if (i < base_energy_history.size())
        entry["base_energy"] = base_energy_history[i];
      if (i < chi_squared_history.size())
        entry["chi_squared"] = chi_squared_history[i];
      
      // On ne garde pas les probabilites par iteration pour ne pas surcharger le log
      // if (i < probs_history.size())
      //   entry["probabilities"] = probs_history[i];

      // On garde les parametres uniquement pour l'iteration finale
      if (i == iter_history.size() - 1) {
        if (i < params_history.size())
          entry["parameters"] = params_history[i];
      }
      history.push_back(entry);
    }
  }
  j["history"] = history;

  // State (Probs)
  /*
  {
    std::lock_guard<std::mutex> lock(graph_mutex);
    j["state"]["probabilities"] = counts_values;

    // Labels
    std::vector<std::string> labels;
    int n_q = 0;
    if (counts_values.size() > 0)
      n_q = (int)std::log2(counts_values.size());
    for (size_t i = 0; i < counts_values.size(); ++i) {
      std::string bitstring = "";
      for (int b = 0; b < n_q; ++b) {
        bitstring += ((i >> (n_q - 1 - b)) & 1) ? "1" : "0";
      }
      labels.push_back(bitstring);
    }
    j["state"]["labels"] = labels;
  }
  */

  // System
  j["system"] = {{"simulator", "VQE Simulator C++ v1.0"},
                 {"num_qubits", num_qubits},
                 {"num_paulis", num_paulis}};

  // 3. Write
  std::string filename = "run_" + filename_ts + ".json";
  std::ofstream o(filename);
  o << std::setw(4) << j << std::endl;

  spdlog::info("Run sauvegarde dans: {}", filename);

  // Save RDMs to a separate file to keep the main log clean
  std::string rdm_filename = "rdms_" + filename_ts + ".json";
  std::ofstream rdm_out(rdm_filename);
  rdm_out << std::setw(4) << final_results.rdms << std::endl;
  spdlog::info("RDMs sauvegardes dans: {}", rdm_filename);

  // --- Warm Start Export ---
  {
    nlohmann::json ws_json;
    nlohmann::json ws_tag;
    std::string ans_type = ansatz_types[ansatz_idx];
    ws_tag["type"] = ans_type;
    ws_tag["num_qubits"] = num_qubits;
    if (ans_type == "HEA") {
      ws_tag["depth"] = hea_depth;
    } else if (ans_type == "UCCSD") {
      // num_electrons is not directly stored in GUI, extract from params_history context
      // For now, we tag with num_qubits which is sufficient for validation
    }
    ws_json["ansatz_tag"] = ws_tag;

    // Get final parameters from history
    std::vector<double> final_params;
    {
      std::lock_guard<std::mutex> lock(graph_mutex);
      if (!params_history.empty()) {
        final_params = params_history.back();
      }
    }
    ws_json["num_params"] = (int)final_params.size();
    ws_json["parameters"] = final_params;

    std::string ws_filename = "warm_start_" + filename_ts + ".json";
    std::ofstream ws_out(ws_filename);
    ws_out << std::setw(4) << ws_json << std::endl;
    spdlog::info("Warm start sauvegarde dans: {}", ws_filename);
  }
}

//------------------------------------------------------------------------------
//     LOGGING
//------------------------------------------------------------------------------

void GUI::Log(const std::string &message) { spdlog::info(message); }
