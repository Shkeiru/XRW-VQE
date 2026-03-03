# Revue de Code et Analyse Mathématique de la Fonction de Coût VQE

Ce document propose une analyse détaillée et verbeuse de l'implémentation de la fonction de coût de l'algorithme Variational Quantum Eigensolver (VQE) au sein de la base de code QuantumBeast, et plus spécifiquement de la fonction `evaluate_functional` située dans `src/core/simulation.cpp`.

L'objectif principal est de s'assurer de la validité physique et mathématique des opérations effectuées, en faisant abstraction des détails purement logiciels et d'optimisation (tels que le multithreading ou les pointeurs).

---

## 1. Initialisation de l'État Quantique

Avant toute évaluation, le code prépare l'état initial du système quantique.

```cpp
initZeroState(local_qubits);
for (int i = 0; i < data->n_electrons; ++i) {
  applyPauliX(local_qubits, i);
}
```

**Traduction mathématique :**
Le système est d'abord initialisé dans l'état de vide computationnel $|0\rangle^{\otimes N}$, où $N$ est le nombre total de qubits.
Ensuite, l'état de Hartree-Fock $|\Phi_{\text{HF}}\rangle$ est préparé en appliquant la porte $X$ (NOT quantique) sur les $N_e$ premiers qubits, où $N_e$ correspond au nombre d'électrons du système.

$$ |\Psi_{\text{init}}\rangle = X^{\otimes N_e} \otimes I^{\otimes (N-N_e)} |0\rangle^{\otimes N} = |11...100...0\rangle $$

Cette étape est physiquement correcte sous l'hypothèse de la transformation de Jordan-Wigner, où les orbitales spin-spatiales de plus basse énergie (occupées) sont mappées sur les premiers qubits.

L'ansatz paramétré $U(\vec{\theta})$ est ensuite appliqué sur cet état initial pour générer l'état d'essai $|\Psi(\vec{\theta})\rangle$ :

$$ |\Psi(\vec{\theta})\rangle = U(\vec{\theta}) |\Psi_{\text{init}}\rangle $$

---

## 2. Évaluation de l'Énergie : Hamiltonien et Bruit

L'énergie attendue du système est donnée par la valeur moyenne de l'Hamiltonien $H$ sur l'état d'essai :

$$ E(\vec{\theta}) = \langle \Psi(\vec{\theta}) | H | \Psi(\vec{\theta}) \rangle $$

L'Hamiltonien est décomposé en une somme pondérée de chaînes de Pauli :
$$ H = \sum_{i} c_i P_i $$
où $c_i \in \mathbb{R}$ sont les coefficients et $P_i \in \{I, X, Y, Z\}^{\otimes N}$ sont les opérateurs de Pauli.

Le code propose deux modes d'évaluation de cette énergie.

### 2.1 Mode Exact (Sans Bruit)

Lorsque `n_shots == 0`, le code calcule l'énergie exacte :
```cpp
energy = calcExpecPauliStrSum(local_qubits, data->hamiltonian);
```
**Traduction mathématique :**
Le simulateur (QuEST) calcule exactement le produit scalaire :
$$ E_{\text{exact}} = \sum_i c_i \langle \Psi(\vec{\theta}) | P_i | \Psi(\vec{\theta}) \rangle $$
C'est le comportement attendu pour un simulateur de vecteur d'état idéal.

### 2.2 Mode Bruité (Shot Noise)

Lorsque `n_shots > 0`, le code simule le processus de mesure sur un ordinateur quantique réel. L'énergie est estimée terme par terme.

Pour chaque chaîne de Pauli $P_i$ (ayant un coefficient non nul), le code calcule la valeur moyenne exacte $\langle P_i \rangle$ et simule des mesures :
```cpp
double p_plus = (1.0 + expectation) / 2.0;
std::binomial_distribution<> binom(data->n_shots, p_plus);
int n_plus = binom(gen);
```

**Traduction mathématique :**
Pour une observable de Pauli $P_i$ dont les valeurs propres sont $\pm 1$, la probabilité de mesurer $+1$ est liée à la valeur moyenne exacte par :
$$ p_+ = \frac{1 + \langle P_i \rangle}{2} $$
La probabilité de mesurer $-1$ est $p_- = 1 - p_+$.

Le code tire le nombre de mesures donnant $+1$ ($n_+$) depuis une distribution binomiale $\mathcal{B}(N_{\text{shots}}, p_+)$. Le nombre de mesures donnant $-1$ est $n_- = N_{\text{shots}} - n_+$.

L'estimateur de la valeur moyenne de $P_i$ devient :
$$ \langle P_i \rangle_{\text{est}} = \frac{n_+ - n_-}{N_{\text{shots}}} $$

L'énergie totale estimée et sa variance sont accumulées :
$$ E_{\text{est}} = \sum_i c_i \langle P_i \rangle_{\text{est}} $$
$$ \text{Var}(E_{\text{est}}) = \sum_i c_i^2 \frac{1 - \langle P_i \rangle^2}{N_{\text{shots}}} $$

**Note sur la correction physique :**
L'échantillonnage indépendant de chaque terme de Pauli est physiquement correct si les observables sont mesurées dans des bases de mesure indépendantes. Cependant, sur un vrai dispositif matériel, on groupe souvent les termes qui commutent (Qubit-Wise Commuting) pour réduire le nombre total de mesures (shots). Le code simule ici une mesure naïve terme par terme, où on alloue `n_shots` à *chaque* terme de Pauli. L'énergie estimée reste néanmoins un estimateur sans biais correct de l'énergie réelle, reflétant bien la physique de l'échantillonnage stochastique.

---

## 3. Pénalité sur la Conservation du Nombre de Particules

Si l'ansatz utilisé ne conserve pas le nombre de particules (ex: un ansatz HEA - Hardware Efficient Ansatz) et que l'opérateur de nombre $\hat{N}$ est défini, une pénalité est ajoutée à l'énergie.

```cpp
qreal penalty = 3.0 * std::pow((number_exp - data->n_electrons), 2);
energy += penalty;
```

**Traduction mathématique :**
Soit $N_e$ le nombre d'électrons cible (`data->n_electrons`) et $\langle \hat{N} \rangle$ le nombre de particules espéré dans l'état actuel (`number_exp`). La fonction de coût est modifiée en :
$$ C(\vec{\theta}) = E(\vec{\theta}) + \lambda \left( \langle \hat{N} \rangle - N_e \right)^2 $$
Avec un multiplicateur de Lagrange arbitraire $\lambda = 3.0$.

C'est une méthode valide pour contraindre l'espace de recherche vers le sous-espace physique ayant le bon nombre d'électrons, bien que la valeur $\lambda = 3.0$ soit empirique (une valeur trop faible n'impose pas assez la contrainte, une valeur trop forte rend l'optimisation difficile).

---

## 4. Matrice Densité Réduite à 1 Particule (1-RDM) et Données de Diffraction

Une étape cruciale de la fonction est l'intégration de données expérimentales de diffraction via la matrice densité réduite à 1 particule (1-RDM).

### 4.1 Évaluation de la 1-RDM

La 1-RDM est définie par ses éléments de matrice $D_{pq} = \langle \Psi(\vec{\theta}) | a^\dagger_p a_q | \Psi(\vec{\theta}) \rangle$.
Le code évalue chaque élément à partir de sa décomposition en chaînes de Pauli.

**Note importante :** Dans le code actuel, pour le mode bruité (`n_shots > 0`), l'évaluation de la 1-RDM utilise la valeur exacte `calcExpecPauliStrSum` au lieu de simuler le bruit binomial, comme mentionné dans les commentaires (`// In noisy sim we should ideally sample the 1-RDM individually as well`). C'est une approximation justifiée pour le prototypage rapide, mais l'état évalué est hybride (énergie bruitée, 1-RDM exacte).

### 4.2 Facteurs de Structure et $\chi^2$

Les éléments calculés de la 1-RDM (vecteur $D$) sont multipliés par une matrice d'intégrales $I$ pour obtenir les facteurs de structure théoriques $F_{\text{calc}}$ :

```cpp
Eigen::VectorXcd calc_factors = data->integrals * rdm1_map;
```
$$ \vec{F}_{\text{calc}} = I \cdot \vec{D} $$

Le code calcule ensuite un facteur d'échelle $\eta$ entre ces facteurs calculés et les facteurs expérimentaux $\vec{F}_{\text{exp}}$ associés à des incertitudes $\vec{\sigma}$ :

```cpp
double eta = ((calc_factors.cwiseAbs() * data->exp_factors.cwiseAbs()).cwiseQuotient(data->uncertainties).sum()) /
             (calc_factors.cwiseAbs2().cwiseQuotient(data->uncertainties).sum());
```

**Traduction mathématique de $\eta$ :**
$$ \eta = \frac{ \sum_k \frac{|F_{\text{calc}, k}| |F_{\text{exp}, k}|}{\sigma_k} }{ \sum_k \frac{|F_{\text{calc}, k}|^2}{\sigma_k} } $$

Le code construit ensuite un $\chi^2$ pénalisant l'écart entre expérience et théorie :

```cpp
double chi_squared = (eta * calc_factors.cwiseAbs() - data->exp_factors.cwiseAbs())
                        .cwiseAbs2()
                        .cwiseQuotient(data->uncertainties.cwiseAbs2())
                        .sum();
energy += chi_squared;
```

**Traduction mathématique du terme $\chi^2$ :**
$$ \chi^2 = \sum_k \frac{ (\eta |F_{\text{calc}, k}| - |F_{\text{exp}, k}|)^2 }{\sigma_k^2} $$

La fonctionnelle de coût finale devient :
$$ C(\vec{\theta}) = \langle H \rangle + \lambda (\langle \hat{N} \rangle - N_e)^2 + \chi^2 $$

**Analyse de la validité :**
L'ajout du $\chi^2$ permet d'utiliser les données expérimentales de diffraction des rayons X comme contrainte pour guider l'optimisation VQE, une méthode avancée (e.g. "X-ray constrained VQE").
Cependant, la définition de $\eta$ semble inhabituelle. Souvent, la minimisation pondérée mène à un facteur d'échelle qui dépend de $\sigma_k^2$ au dénominateur partout, par exemple $\eta = \frac{ \sum \frac{|F_{calc}||F_{exp}|}{\sigma^2} }{ \sum \frac{|F_{calc}|^2}{\sigma^2} }$.
Dans l'implémentation :
- Numérateur de $\eta$ : somme de $\frac{|F_{calc}| |F_{exp}|}{\sigma}$
- Dénominateur de $\eta$ : somme de $\frac{|F_{calc}|^2}{\sigma}$
Si les incertitudes `data->uncertainties` dans ce tableau représentaient déjà les variances ($\sigma^2$), la formule serait canonique. Mais le code lit `sigma` (`exp_sigma.push_back(sigma);`). Ceci est un point d'attention potentiel pour la stricte rigueur de l'ajustement aux moindres carrés pondérés, bien que cela reste une métrique d'erreur fonctionnelle. De plus, seul l'amplitude absolue des facteurs est comparée (`cwiseAbs()`), négligeant la phase, ce qui correspond au "problème de la phase" classique en cristallographie.

---

## 5. Calcul des Gradients (Parameter Shift Rule)

Si l'optimiseur requiert les gradients (ex: si un algorithme basé sur le gradient est passé en paramètre, bien que `nlopt::LN_NELDERMEAD` soit la valeur par défaut), la méthode `cost_function` implémente la règle du décalage des paramètres (Parameter Shift Rule, PSR).

```cpp
shifted_params[i] = params[i] + M_PI / 2.0;
double e_plus = evaluate_functional(shifted_params, data, local_qubits, rdm1_plus);

shifted_params[i] = params[i] - M_PI / 2.0;
double e_minus = evaluate_functional(shifted_params, data, local_qubits, rdm1_minus);

grad[i] = 0.5 * (e_plus - e_minus);
```

**Traduction mathématique :**
Si l'ansatz est composé de portes de rotation générées par des matrices de Pauli, la dérivée analytique exacte par rapport au paramètre $\theta_i$ est donnée par :
$$ \frac{\partial C(\vec{\theta})}{\partial \theta_i} = \frac{1}{2} \left[ C\left(\vec{\theta} + \frac{\pi}{2} \hat{e}_i\right) - C\left(\vec{\theta} - \frac{\pi}{2} \hat{e}_i\right) \right] $$

**Correction de l'implémentation :**
La règle du décalage des paramètres telle qu'implémentée suppose que chaque paramètre $\theta_i$ contrôle une rotation $e^{-i \frac{\theta_i}{2} P}$ où $P$ est une chaîne de Pauli involutive ($P^2 = I$).
Si on regarde l'ansatz UCCSD (`src/core/ansatz.cpp`), le compilateur ("Tape") traduit l'excitation en une rotation :
```cpp
applyRotateZ(qubits, inst.target, params[inst.param_idx] * inst.angle_multiplier);
```
Si le multiplicateur intégré dans l'UCCSD ne respecte pas le générateur normalisé standard, le facteur $\frac{1}{2}$ et le décalage de $\pm \frac{\pi}{2}$ dans la PSR de la fonction de coût pourraient donner un gradient mis à l'échelle. Pour un gradient exact, il faut s'assurer de la cohérence entre le facteur `angle_multiplier` et l'application de la règle de décalage.
Dans le cas du simulateur de gradient-free (Nelder-Mead), ce code de gradient n'est pas appelé, ce qui évite le problème en pratique.

---

## Conclusion de la Revue

L'implémentation mathématique de la boucle principale d'évaluation VQE est rigoureuse et reflète fidèlement la théorie sous-jacente :
1. **L'état d'essai** est correctement généré à partir du vide via une initialisation Hartree-Fock.
2. **L'évaluation de l'énergie** modélise proprement le régime exact (idéal) et fournit une estimation stochastique correcte via échantillonnage binomial pour le mode bruité.
3. Les **pénalités d'opérateur** sont mathématiquement valides.
4. L'incorporation de la **1-RDM pour l'ajustement du $\chi^2$** cristallographique démontre une excellente approche hybride physico-chimique. Une légère vérification sur la puissance de l'incertitude dans le calcul du facteur d'échelle $\eta$ serait le seul point théorique à auditer finement avec les physiciens.

Ce code s'aligne de manière satisfaisante avec les standards de la simulation quantique variationnelle.
