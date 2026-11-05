# PLAN — Repo 5 : structures par terme, carry, stockage, spreads (énergie)

> Phase 0 d'un repo créé de zéro, rédigée le 2026-10-12. Tient lieu d'`AUDIT.md` pour ce repo :
> les risques sont classés par gravité au §3. À relire avant chaque séance avec `PROGRESS.md`.
> Nom retenu : repo `Commodities-curves-and-carry`, package `energycurves`.

---

## 0. Objet

Phrase README (provisoire, à figer en phase 5) :
*Term-structure analytics, a carry backtest and a natural-gas storage valuation on NYMEX energy
futures (WTI, Henry Hub, RBOB, heating oil), built only on public EIA/FRED data.*

Ce que le repo doit prouver en 90 s à une trading house, un énergéticien ou un courtier :
1. mécanique futures maîtrisée : expiration, roll, rendement à contrat constant ;
2. valorisation d'un actif physique par optimisation, séparation intrinsèque / extrinsèque ;
3. hygiène de backtest : pas de look-ahead, coûts, puissance statistique déclarée.

---

## 1. Données — constat vérifié le 2026-10-12

| Série | Source | Historique quotidien | Statut |
|---|---|---|---|
| WTI Cushing, contrats 1–4 | EIA (clôture NYMEX 14h30) | C1, C3 : 1983 · C2, C4 : 1985 → **2024-04-05** | Arrêtée |
| Henry Hub, contrats 1–4 | EIA | C1–C3 : 1994 · C4 : 1993 → **2024-04-05** | Arrêtée |
| RBOB NY Harbor, contrats 1–4 | EIA | 2005 → **2024-04-05** | Arrêtée |
| No. 2 heating oil NY Harbor, contrats 1–4 | EIA | C1, C3 : 1980 · C2, C4 : 1994 → **2024-04-05** | Arrêtée |
| Spots : WTI Cushing, Brent, NYH gasoline, NYH heating oil, NYH ULSD, Henry Hub | EIA | 1986 / 1987 / 1986 / 1986 / 2006 / 1997 → en cours | Active |
| T-bill 3 mois (`DTB3`) | FRED | — | Active |

L'EIA affiche sur ses pages futures : « Futures prices after April 5, 2024, are not available ».

**Conséquences de conception**

- **Échantillon figé au 2024-04-05.** Avantage : jeu immuable, versionné intégralement dans
  `data/sample/` (taille estimée < 5 Mo en parquet, à mesurer en S1) → reproductibilité sans clé API.
- **Univers = 4 contrats, pas 5–6.** L'EIA ne publie aucun futures Brent ; la section propane est
  vide. CL, RB et HO sont fortement corrélés : l'univers contient en pratique ~2 paris
  indépendants (complexe pétrolier, gaz). Limite à écrire noir sur blanc dans le README.
- **Échantillon commun aux 4 actifs** : 2005 → 2024-04 ≈ 18,5 ans (borne : RBOB).
- **Extension post-2024 (optionnelle)** : contrats individuels via `yfinance` — *non vérifié*.
  Spike limité à 30 min en S1 ; en cas d'échec, pas d'extension et limite écrite.
- **Module 3** : 4 contrats = 4 mois de courbe, insuffisant pour une année de stockage. Deux
  courbes NG sur 12 mois :
  - (a) *primaire* : snapshot de la courbe NYMEX complète via `yfinance`, figé avec date
    d'extraction (si le spike réussit) ;
  - (b) *toujours implémentée* : courbe saisonnière moyenne reconstruite par chaînage des
    spreads adjacents EIA (moyenne de $\ln F^{(k+1)} - \ln F^{(k)}$ par paire de mois de
    livraison, 12 paires chaînées). Sert de repli, de test hors ligne et de robustesse.

**Extraction** : fichiers « Download Series History » de l'EIA, sans clé API
(`https://www.eia.gov/dnav/ng/xls/NG_PRI_FUT_S1_D.xls`, équivalent `pet/xls/PET_PRI_FUT_S1_D.xls`
et `PET_PRI_SPT_S1_D.xls` — URL pétrole à confirmer en S1). Format `.xls` → `xlrd`.
`data/sample/MANIFEST.json` : pour chaque fichier, URL, date d'extraction UTC, sha256, plage, unité.

**Ruptures à documenter dans `docs/DATA.md`**

- Unités : CL en \$/bbl ; RB, HO en \$/gal (×42 → \$/bbl) ; NG en \$/MMBtu.
- WTI contrat 1 négatif le 2020-04-20 : $\ln$ indéfini.
- Heating oil : spécification NYMEX passée au ULSD en 2013 (à confirmer), libellé EIA inchangé.
- RBOB remplace le RFG (série RFG arrêtée en 2006).

**Règles d'expiration (définitions EIA)**

- Essence et heating oil : dernier jour ouvré du mois précédant la livraison.
- Brut : 3e jour ouvré avant le 25 du mois précédant la livraison (4e si le 25 est chômé).
- Gaz naturel : 3e jour ouvré avant le 1er du mois de livraison (règle CME, à confirmer).
- Après expiration, le contrat 1 devient le mois de livraison suivant ; les contrats 2–4 suivent.

---

## 2. Arborescence

```
Commodities-curves-and-carry/
├── CLAUDE.md  PLAN.md  PROGRESS.md  README.md
├── pyproject.toml              # deps épinglées ; extras [dev], [fetch] (yfinance)
├── .github/workflows/ci.yml    # ruff, mypy, pytest --cov
├── src/energycurves/
│   ├── data/
│   │   ├── registry.py         # catalogue : id série, produit, contrat, unité, source
│   │   ├── eia.py              # parsing xls → DataFrame long (date, produit, contrat, prix)
│   │   ├── calendar.py         # expirations NYMEX, jours ouvrés, τ_k(t)
│   │   └── stitch.py           # rendements à contrat constant, détection de roll
│   ├── curves.py               # M1 : pentes, régimes, roll yield
│   ├── carry/
│   │   ├── signals.py          # pente brute et désaisonnalisée (fenêtre expansive)
│   │   ├── backtest.py         # TS, XS, long-only ; coûts
│   │   └── metrics.py          # rendement, vol, Sharpe, t-stat, MDD, turnover
│   ├── storage/
│   │   ├── contract.py         # dataclass figée StorageContract
│   │   ├── curve.py            # courbes (a) snapshot et (b) chaînage saisonnier
│   │   ├── intrinsic.py        # LP scipy.optimize.linprog (HiGHS)
│   │   ├── forward_model.py    # dynamique 2 facteurs : calibration, simulation exacte
│   │   └── rolling.py          # rolling intrinsic
│   └── spreads/                # M4 (option)
│       ├── schwartz.py  kirk.py  montecarlo.py
├── tests/                      # un fichier par module + test_regression.py
├── data/
│   ├── raw/                    # .gitignore
│   └── sample/                 # parquet figés + MANIFEST.json
├── scripts/
│   ├── fetch_data.py           # re-télécharge, vérifie sha256
│   └── run_results.py          # régénère toutes les tables et figures du README
├── notebooks/demo.ipynb        # < 2 min
└── docs/
    ├── DATA.md
    └── figures/
```

Conventions : fonctions pures sur `DataFrame`/`ndarray` ; I/O limitée à `energycurves/data/` et
`scripts/` ; paramètres en `dataclass(frozen=True)` ; aléa par `np.random.Generator` passé en argument.

---

## 3. Risques classés par gravité

### Critique — fausse les résultats si ignoré

**R1. Raccord des contrats.** Un `pct_change` sur la série « contrat 1 » brute incorpore le saut
de roll, c'est-à-dire le roll yield lui-même : le backtest carry devient mécaniquement biaisé.
Parade : rendement à contrat constant (§4, M2) ; calendrier d'expiration validé empiriquement.

**R2. Saisonnalité.** Pour NG, RBOB et HO, la pente entre contrats adjacents est dominée par la
saisonnalité (hiver, spécification été/hiver de l'essence). Un « signal carry » brut est alors un
pari saisonnier. Parade : signal désaisonnalisé ; publier brut et désaisonnalisé côte à côte.

**R3. Look-ahead.** Moyenne saisonnière, vol de scaling, calibration : uniquement des fenêtres
expansives ou glissantes strictement passées. Test dédié (perturber les données après $t$ ne
modifie pas les positions en $t$).

**R4. Modèle à 1 facteur pour le stockage.** Corrélation 1 entre maturités → spreads calendaires
quasi déterministes → valeur extrinsèque proche de 0 par construction. Parade : 2 facteurs.

### Élevé

**R5. Identification.** $\tau$ ne couvre que ~0–4 mois : $\kappa$ et $\sigma_l$ sont mal
identifiés ; la vol au-delà de 4 mois est extrapolée par le modèle. Parade : grille de
sensibilité de la valeur RI à $(\kappa, \sigma_l, \rho)$ publiée.

**R6. Puissance statistique.** Sur ~18,5 ans communs, un Sharpe n'est distinguable de 0 à
$t = 2$ que s'il dépasse $2/\sqrt{18{,}5} \approx 0{,}47$. Publier les t-stats, ne rien affirmer au-delà.

**R7. Prix négatif WTI (2020-04-20).** Parade : positions sur le contrat 2 ; signal calculé sur
contrats 2–3 si le contrat 1 est ≤ 0 ; test explicite.

**R8. Kirk à 3 jambes.** Kirk est une approximation à 2 actifs. Parade : panier de produits
approché lognormal par 2 moments, puis Kirk ; comparaison au Monte Carlo *sous le même modèle*,
sinon l'écart mélange erreur de modèle et erreur d'approximation.

### Moyen

**R9. Jours fériés NYMEX** ≠ calendrier fédéral US (Good Friday, fermetures exceptionnelles).
Parade : calendrier fédéral + Good Friday, validé contre les rolls détectés dans les données.

**R10. Rupture de spécification HO (2013)** : documenter, tester la stabilité avant/après.

**R11. Snapshot yfinance** : disponibilité et conditions d'utilisation incertaines ; fichier
minimal, documenté ; repli (b) toujours disponible.

**R12. Budget.** 15–20 h pour M1–M3 ne couvre pas le standard (tests, CI, README, livraison).
Estimation : ~23,5 h pour le MVP (§5).

### Faible

**R13.** Format `.xls` ancien (dépendance `xlrd`). **R14.** Révisions EIA : sans effet, données figées.

---

## 4. Spécification des modules

### M1 — Structures par terme

Entrées : panel $F_t^{(k)}$, $k = 1..4$ ; $\tau_k(t)$ issu de `calendar.py`.

$$
y_t^{(i,j)} = \frac{\ln F_t^{(i)} - \ln F_t^{(j)}}{\tau_j(t) - \tau_i(t)}, \qquad y > 0 \iff \text{backwardation}
$$

Sorties : $y^{(1,2)}$, $y^{(2,3)}$, $y^{(1,4)}$ ; régime contango / plat / backwardation avec bande
morte $|y| < \varepsilon$ (paramètre, défaut 1 %/an) ; par produit : roll yield moyen annualisé,
part du temps en backwardation, découpage par sous-période.

### M2 — Carry

Positions sur le contrat 2 (évite le mois de livraison et la séance d'expiration du contrat 1).

Rendement à contrat constant : si le contrat 1 a expiré entre $t-1$ et $t$, le contrat qui était
$k$ devient $k-1$ :

$$
r_t = \frac{F_t^{(k-1)}}{F_{t-1}^{(k)}} - 1 \ \text{(jour de roll)}, \qquad
r_t = \frac{F_t^{(k)}}{F_{t-1}^{(k)}} - 1 \ \text{(sinon)}
$$

Signal : $s_t = y_t^{(1,2)}$ (carry d'un contrat 2 qui glisse vers la position 1 à courbe
inchangée). Désaisonnalisé :

$$
\tilde s_t = s_t - \frac{1}{|U_t|} \sum_{u \in U_t} s_u, \qquad U_t = \{u < t : m(u) = m(t)\}, \ \text{minimum 3 ans d'historique}
$$

Stratégies (rebalancement mensuel au 10e jour ouvré, à l'écart de toutes les expirations) :
- **TS** : $w_i = \operatorname{sign}(\tilde s_i)\,(\sigma^*/\hat\sigma_i)/N$, $\hat\sigma$ EWMA 60 j, $\sigma^* = 10\,\%$/an ;
- **XS** : rang sur $\tilde s$, long 2 / short 2, pondération inverse-vol, somme nulle ;
- **Référence** : long-only à risque égal (sépare la prime passive du signal).

Métriques : rendement et vol annualisés, Sharpe (rendements excédentaires), t-stat
$\approx SR\sqrt{\text{années}}$, drawdown maximal, turnover, taux de réussite. Sensibilité :
coût proportionnel au notionnel échangé, grille {0, 2, 5, 10, 20} bp. Toutes les variantes
publiées en brut et désaisonnalisé.

### M3 — Stockage de gaz

Contrat (hypothèses illustratives, paramétrables) : année avril 2027 – mars 2028 ; capacité
$\bar V = 1$ (valeur en \$/MMBtu de capacité) ; injection ≤ $\bar V/6$ par mois ; soutirage
≤ $\bar V/3$ par mois ; $V_0 = V_{12} = 0$ ; coûts $c_i = c_w = 0{,}02$ \$/MMBtu ; actualisation
au `DTB3` de la date du snapshot ; pas de ratchets.

$$
\begin{aligned}
\max_{i,w}\ & \sum_{m=1}^{M} D_m \left[ F_m (w_m - i_m) - c_i\, i_m - c_w\, w_m \right] \\
\text{s.c.}\ & V_m = V_{m-1} + i_m - w_m,\quad 0 \le V_m \le \bar V,\quad 0 \le i_m \le \bar I_m,\quad 0 \le w_m \le \bar W_m
\end{aligned}
$$

Solveur : `scipy.optimize.linprog(method="highs")`.

Dynamique de courbe sous $\mathbb{Q}$ (forwards martingales), 2 facteurs :

$$
\frac{dF(t,T)}{F(t,T)} = \sigma_s e^{-\kappa (T-t)}\, dW_t^s + \sigma_l\, dW_t^l, \qquad d\langle W^s, W^l\rangle_t = \rho\, dt
$$

Calibration : covariance empirique des $\Delta \ln F^{(k)}$ hors jours de roll (fenêtre par défaut
2014-04 → 2024-04, sensibilité à la fenêtre), moindres carrés non linéaires sur

$$
\Sigma_{ij} = \sigma_s^2 e^{-\kappa(\tau_i+\tau_j)} + \sigma_l^2 + \rho\,\sigma_s\sigma_l\left(e^{-\kappa\tau_i} + e^{-\kappa\tau_j}\right)
$$

Simulation exacte (log-normale, variance intégrée par pas mensuel), 2 000 chemins, seed fixée.

Rolling intrinsic : à chaque début de mois, ré-optimiser le LP sur les mois restants depuis le
stock courant, exécuter l'action du mois, verrouiller par forwards. Sans coûts de re-couverture,
le P&L RI est ≥ intrinsèque **chemin par chemin** (le plan couvert reste réalisable) : c'est un test.

Sorties : intrinsèque $I$ ; RI moyen et IC 95 % ; part extrinsèque $(RI - I)/I$ ; sensibilités.
Limites : RI = borne inférieure de la valeur optimale ; pas de vol saisonnière ; granularité
mensuelle (pas d'optionalité spot intra-mois) ; pas de ratchets ; pas de base Henry Hub / hub physique.

### M4 (option) — Spreads

- Schwartz 1 facteur par jambe (CL, RB, HO) : $\sigma_F(t,T) = \sigma e^{-\kappa(T-t)}$ ;
  calibration par régression $\ln \hat\sigma(\tau) = \ln\sigma - \kappa\tau$ sur les contrats 1–4
  (effet Samuelson) ; corrélations inter-jambes sur $\Delta\ln F^{(2)}$.
- Variance intégrée (option d'échéance $T_o$ sur futures d'échéance $T_F \ge T_o$) :
  $v^2 = \sigma^2 e^{-2\kappa(T_F-T_o)}\,(1 - e^{-2\kappa T_o})/(2\kappa)$ ; covariance croisée
  analogue avec $\kappa_i + \kappa_j$.
- Payoff 3-2-1 en \$/bbl : $\left[\tfrac{1}{3}(2\cdot 42\, RB_T + 42\, HO_T) - CL_T - K\right]^+$.
- Kirk (2 jambes, $D$ facteur d'actualisation) :

$$
C \approx D\left[F_1 N(d_1) - (F_2+K) N(d_2)\right],\quad
\sigma_K^2 = \sigma_1^2 - 2\rho\sigma_1\sigma_2\frac{F_2}{F_2+K} + \sigma_2^2\left(\frac{F_2}{F_2+K}\right)^2
$$

$$
d_1 = \frac{\ln\!\big(F_1/(F_2+K)\big) + \tfrac12 \sigma_K^2 T}{\sigma_K\sqrt T},\qquad d_2 = d_1 - \sigma_K\sqrt T
$$

- 3 jambes : panier $B = \tfrac13(2\cdot42\,RB + 42\,HO)$ approché lognormal par appariement de
  $\mathbb E[B]$ et $\mathbb E[B^2]$ ; corrélation $B$–CL par appariement de $\mathbb E[B\cdot CL]$ ; puis Kirk.
- Monte Carlo : jambes conjointement lognormales sous le modèle → tirage direct des valeurs
  terminales, sans discrétisation ; antithétiques ; $N = 10^6$ ; erreur standard publiée.
- Résultat : écart Kirk − MC (\$/bbl et % du prix MC) sur grille strike × maturité, décomposé en
  erreur Kirk pure (spread HO − CL, 2 jambes) et erreur ajoutée par l'approximation de panier.

---

## 5. Plan des phases adapté (repo créé de zéro)

Adaptation du standard : la phase 2 (tests) est intégrée à chaque module ; phases 3–6 inchangées.

| Séance | Phase | Contenu | Critère de fin | h |
|---|---|---|---|---|
| S1 | 1 | Squelette, `pyproject.toml`, config ruff/mypy, `fetch_data.py` → parquet + MANIFEST ; spike yfinance (30 min max) | `pip install -e .` OK ; parquet chargés ; décision (a)/(b) notée | 2,5 |
| S2 | 1 | `calendar.py`, `stitch.py` + tests | rolls calendaires = rolls détectés ≥ 99 % ; tests verts | 2,5 |
| S3 | M1 | Pentes, régimes, table et figure | table roll yields produite par script | 2 |
| S4 | M2 | Signaux brut / désaisonnalisé, backtests TS, XS, référence | P&L cumulés | 2,5 |
| S5 | M2 | Métriques, coûts, tests anti-look-ahead | table carry + figure Sharpe vs coût | 1,5 |
| S6 | M3 | Courbes (a)/(b), LP intrinsèque + tests | valeur intrinsèque + plan optimal | 2,5 |
| S7 | M3 | Calibration 2 facteurs, simulation + tests | paramètres + test martingale | 2,5 |
| S8 | M3 | Rolling intrinsic, sensibilités | table I vs RI | 2 |
| S9 | 3–4 | CI + badge ; `run_results.py` complet | CI verte ; tout régénéré | 2 |
| S10 | 5 | README anglais, ordre imposé | relu contre ce plan | 2 |
| S11 | 6 | Notebook < 2 min, tag v1.0, notes de version, post LinkedIn | repo épinglé | 1,5 |
| | | **Total MVP** | | **≈ 23,5** |
| S12–S14 | M4 | Schwartz, Kirk, MC, tests → v1.1 | table Kirk vs MC | ≈ 7 |

Point de coupe : si le cumul dépasse 26 h à la fin de S8, livrer v1.0 sans la grille de
sensibilité RI. En cas de dérapage, réduire M2 au carry TS plutôt que sacrifier M3, qui est le
module le plus différenciant pour les trading houses et énergéticiens.

---

## 6. Résultats cibles (phase 4) — formats uniquement, aucune valeur avant exécution

| # | Contenu | Format |
|---|---|---|
| 1 | Roll yield moyen annualisé, part du temps en backwardation, par produit et sous-période | Table |
| 2 | Carry TS / XS / référence, brut vs désaisonnalisé : rendement, vol, Sharpe, t-stat, MDD, turnover | Table + P&L cumulé + Sharpe vs coût |
| 3 | Stockage : $I$, RI moyen [IC 95 %], part extrinsèque, sensibilités | Table + plan optimal sur la courbe |
| 4 | (M4) Kirk vs MC sur grille strike × maturité | Table + figure d'erreur |

Chaque résultat est interprété en une phrase, rédigée après exécution de `run_results.py`.

---

## 7. Références (citées de mémoire : volumes et pages à vérifier avant le README)

- Fama, E. & French, K. (1987). Commodity futures prices: some evidence on forecast power, premiums, and the theory of storage. *Journal of Business*.
- Erb, C. & Harvey, C. (2006). The strategic and tactical value of commodity futures. *Financial Analysts Journal*.
- Gorton, G., Hayashi, F. & Rouwenhorst, K. G. (2013). The fundamentals of commodity futures returns. *Review of Finance*.
- Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018). Carry. *Journal of Financial Economics*.
- Schwartz, E. (1997). The stochastic behavior of commodity prices. *Journal of Finance*.
- Margrabe, W. (1978). The value of an option to exchange one asset for another. *Journal of Finance*.
- Kirk, E. (1995). Correlation in the energy markets. In *Managing Energy Price Risk*, Risk Publications.
- Carmona, R. & Durrleman, V. (2003). Pricing and hedging spread options. *SIAM Review*.
- Secomandi, N. (2010). Optimal commodity trading with a capacitated storage asset. *Management Science*.
- Lai, G., Margot, F. & Secomandi, N. (2010). An approximate dynamic programming approach to benchmark practice-based heuristics for natural gas storage valuation. *Operations Research*.
- Données : EIA, *NYMEX Futures Prices* (pet_pri_fut_s1_d), *Natural Gas Spot and Futures Prices* (ng_pri_fut_s1_d), *Spot Prices* (pet_pri_spt_s1_d) ; FRED `DTB3`.
