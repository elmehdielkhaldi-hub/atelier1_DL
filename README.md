# Lab 1 — Deep Learning avec PyTorch : Régression & Classification multi-classe

**Master SITBD — FSTT, Université Abdelmalek Essaâdi** · Module *Deep Learning* · Pr. Lotfi ELAACHAK
**Étudiant :** El Mehdi El Khaldi

## Objectif
Se familiariser avec **PyTorch** pour des tâches de **régression** et de **classification** en construisant des architectures **DNN/MLP**. Le lab couvre l'EDA, la recherche d'hyper-paramètres par GridSearch, l'analyse des courbes d'apprentissage et la régularisation.

## Contenu du dépôt
| Fichier | Description |
|---|---|
| `Partie1_Regression_NYSE.ipynb` | Partie 1 : régression sur le dataset NYSE (exécuté sur Google Colab, GPU T4) |
| `Partie2_Classification_Maintenance.ipynb` | Partie 2 : classification multi-classe, maintenance prédictive *(à venir)* |
| `requirements.txt` | Dépendances Python |

**Outils :** Python, PyTorch, scikit-learn, skorch (pont PyTorch ↔ `GridSearchCV`), imbalanced-learn, pandas, matplotlib, seaborn, Google Colab.

**Exécution :** ouvrir le notebook dans Google Colab, puis *Exécution → Tout exécuter*. Le dataset est téléchargé automatiquement par la première cellule.

---

## Partie 1 — Régression (NYSE)
**Dataset :** [NYSE — S&P 500 prices](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv` (851 264 lignes, 501 actions, 2010–2016). On utilise les prix ajustés des *splits* pour éviter de fausses chutes de prix.

### 1. Exploratory Data Analysis
- Aucune valeur manquante, aucun doublon, aucune incohérence (open/close toujours compris entre low et high). Seules 21 lignes ont un volume nul : elles sont supprimées.
- Les prix et les volumes sont très asymétriques (échelle log). Le rendement journalier est centré sur 0 (σ ≈ 1,5 %), avec des queues épaisses.
- **Le piège :** `open`, `close`, `low` et `high` sont corrélés à **0,9999**. La baseline « close = open » obtient déjà un R² ≈ 0,9997 : prédire le prix brut ne mesurerait rien.
- **Reformulation :** on prédit le **rendement intrajournalier** `ret = (close − open)/open × 100`, puis on reconstruit le prix de clôture.

### 2. Architecture DNN
- **Features relatives :** `hi`, `lo` (amplitude de la journée), `gap` (saut d'ouverture), `prev_ret` (rendement de la veille), `logv_rel` (volume anormal). Ces deux dernières n'utilisent que le passé, pour éviter toute fuite d'information.
- **Split temporel** : train 2010–2014 / validation 2015 / test 2016. Les extrêmes sont écrêtés (0,1 %–99,9 %) et les features standardisées (`StandardScaler`), avec des seuils et un scaler ajustés sur le train uniquement.
- **MLP paramétrable** (`nn.Linear` + activation, initialisation Kaiming/Xavier, sortie linéaire), loss **MSE**, boucle `zero_grad → forward → loss → backward → step`.
- **Sanity check :** le réseau apprend par cœur 64 exemples (loss 4,87 → 0,0002), ce qui valide le pipeline.

### 3. GridSearch des hyper-paramètres
`GridSearchCV` + **skorch** + **`TimeSeriesSplit`** (validation toujours postérieure à l'entraînement), sur un sous-échantillon de 60 000 lignes : 48 combinaisons × 3 folds = 144 entraînements (≈ 12 min sur GPU).

| Hyper-paramètre | Valeurs testées | Meilleure valeur |
|---|---|---|
| Architecture | (64, 32), (128, 64, 32) | **(128, 64, 32)** |
| Activation | ReLU, Tanh | **Tanh** |
| Optimiseur | SGD + Momentum, RMSprop, Adam | **SGD + Momentum** |
| Learning rate | 1e-3, 1e-2 | **1e-2** |
| Epochs | 10, 20 | **10** |

**Meilleur R² (CV) = 0,805.** Les meilleures configurations sont très proches (0,804–0,805), avec des écarts du même ordre que l'écart-type entre folds (≈ 0,003–0,005). Plusieurs réglages sont donc équivalents. SGD avec un petit learning rate (1e-3) est le moins bon : il n'a pas le temps de converger.

### 4. Courbes Loss/Epochs et Accuracy/Epochs
En régression, l'accuracy n'existe pas. On utilise son équivalent, le **R²** (1 = parfait, 0 = pas mieux que la moyenne).

| Modèle (test 2016) | R² rendement | RMSE prix ($) |
|---|---|---|
| Baseline naïve (close = open) | −0,002 | 1,699 |
| Régression linéaire | 0,800 | 0,745 |
| **DNN (MLP)** | **0,804** | **0,728** |

**Interprétation :**
- **Convergence rapide :** la loss baisse surtout dans les premières epochs, puis les courbes forment un plateau bruité.
- **Léger overfitting :** le R² train continue de monter (≈ 0,807 → 0,815) alors que les R² validation et test plafonnent autour de 0,80. Le réseau commence à apprendre des détails propres à la période 2010–2014.
- **Loss test plus haute que la loss train à R² égal :** la volatilité de 2016 est différente. La MSE dépend de la variance de la période, le R² la normalise.
- **Gain réel mais modeste :** le DNN divise l'erreur sur le prix par plus de 2 par rapport à la baseline naïve, mais ne fait qu'à peine mieux que la régression linéaire. La relation est en grande partie linéaire : le Deep Learning n'est pas magique sur des données tabulaires simples.

### 5. Régularisation
Techniques comparées au modèle de base : **L2** (weight decay), **L1** (pénalité sur les poids), **Dropout**, **Early stopping**, **BatchNorm + Dropout + L2 + Early stopping**. Une expérience contrôlée complète la comparaison : petit dataset (2 000 lignes) + gros réseau (256-256-128), pour provoquer l'overfitting et voir l'effet de chaque technique.

> 🔄 *Résultats chiffrés à compléter après la fin de l'exécution (tableau comparatif dans le notebook).*

**Interprétation attendue :**
- **Dataset complet (≈ 600 000 lignes) :** gains faibles. Le modèle de base n'overfitte presque pas, il y a donc peu de variance à réduire, et trop de régularisation ajoute du biais. L'early stopping atteint un score équivalent en moins d'epochs.
- **Petit dataset :** sans régularisation, la loss test remonte pendant que la loss train baisse (overfitting net). L2, L1 et Dropout resserrent l'écart train/test, et l'early stopping coupe près du minimum de la validation.

---

## Partie 2 — Classification multi-classe (maintenance prédictive)
*(à venir)*

---

## Synthèse : ce que j'ai appris
- **L'EDA passe avant le modèle.** Elle a révélé que prédire le prix brut était trivial (corrélation de 0,9999), et elle a changé la formulation du problème. Un score n'a de sens que comparé à une **baseline**.
- **Éviter les fuites de données :** split **temporel** pour des séries chronologiques, scaler et seuils ajustés sur le **train uniquement**, features construites uniquement à partir du passé.
- **PyTorch en pratique :** `Dataset`/`DataLoader`, mini-batches, `nn.Module` paramétrable, boucle d'entraînement (`zero_grad → forward → loss → backward → step`), modes `train()`/`eval()`, autograd.
- **Optimisation :** rôle du learning rate et différences entre SGD, Momentum, RMSprop et Adam. Une **recherche systématique** (GridSearch + validation croisée adaptée au temps) vaut mieux qu'un choix au hasard. Quand les écarts sont minimes, on préfère la configuration la plus simple et la plus stable.
- **Lire les courbes d'apprentissage** pour diagnostiquer l'underfitting, l'overfitting et le changement de distribution entre périodes.
- **La régularisation n'est pas automatique :** elle réduit la variance au prix d'un peu de biais. Elle est utile quand le modèle overfitte (peu de données, beaucoup de paramètres), et presque inutile sinon. **On diagnostique d'abord, on régularise ensuite.**
- **Esprit critique :** sur des données tabulaires simples, un DNN n'apporte pas forcément plus qu'une régression linéaire. Il faut toujours le comparer à un modèle simple.
- **Côté pratique :** sur GPU, le goulot d'étranglement est souvent le chargement des données (DataLoader), pas le calcul.
