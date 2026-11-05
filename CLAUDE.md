# CLAUDE.md — commodities-curves-and-carry (package `energycurves`)

Début de séance : lire ce fichier, `PLAN.md`, `PROGRESS.md`. Fin de séance : mettre à jour `PROGRESS.md`.
Une séance du `PLAN.md` §5 par séance. Plan validé par l'utilisateur avant toute refonte importante.

## Contexte auteur

- Tancrède : MSc Finance ESSEC (Financial Markets), ingénieur JUNIA HEI (finance quantitative).
  Cible : front-office S&T, structuring, commodities.
- Budget : 6–8 h/semaine en soirée, séances de 1,5 à 2,5 h.
- **Interdit absolu** : toute donnée, méthode ou information liée au stage en cours. Données publiques uniquement.

## Objectif

Un recruteur quant/trading comprend en 90 s ce que fait le repo et ce qu'il prouve, et relance tout en 2 commandes :

```
pip install -e ".[dev]"
python scripts/run_results.py
```

## Règles non négociables

1. **Aucun chiffre inventé.** Toute valeur du README, du notebook ou de `docs/` sort de
   `scripts/run_results.py` exécuté. Jamais de valeur « attendue », « typique » ou estimée.
2. **Données.** Résultats calculés uniquement depuis `data/sample/` (versionné). `scripts/fetch_data.py`
   re-télécharge et vérifie le sha256 du `MANIFEST.json`. Toute nouvelle série : URL, date d'extraction
   UTC, unité, plage → `MANIFEST.json` et `docs/DATA.md`.
3. **Pas de look-ahead.** Tout estimateur utilisé en $t$ (moyenne saisonnière, vol, calibration) n'utilise
   que des données strictement antérieures à $t$. Test obligatoire.
4. **Rendements futures à contrat constant.** Jamais de `pct_change` sur une série « contrat k » brute.
5. **Code.** Fonctions pures, type hints, docstring avec formule et référence. Aucun chemin absolu.
   I/O limitée à `src/energycurves/data/` et `scripts/`. Aléa : `np.random.Generator` passé en argument ;
   seeds fixées dans les scripts.
6. **Commits atomiques**, préfixes `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:`.
   Push direct sur `main` (instructions du projet Claude, qui priment sur le brief SQL) ; si les
   tests échouent : branche `fix/<date>`. Jamais de force push ni de réécriture d'historique.
   Auteur : Tancrède Geli <282603140+tancredeg@users.noreply.github.com>.
7. Réponses denses ; signaler explicitement hypothèses et incertitudes.

## Outils

- Python 3.12 (CI), dépendances épinglées dans `pyproject.toml`.
  Base : numpy, pandas, scipy, matplotlib, pyarrow, duckdb. Extra `[fetch]` : xlrd, yfinance
  (seulement pour reconstruire l'échantillon). Extra `[dev]` : pytest, pytest-cov, ruff, mypy,
  pandas-stubs, nbconvert.
- ruff (lint + format) ; mypy `--strict` sur `src/`.
- GitHub Actions : lint + mypy + tests à chaque push ; badge dans le README.
- Couverture ≥ 80 % sur `curves`, `carry`, `storage`, `spreads`.

## Couche SQL (DuckDB) — brief `brief_github5_sql.md`

- Python télécharge et trace ; SQL stocke, contrôle et calcule les indicateurs.
- `sql/00_schema.sql` → `10_load_*.sql` (idempotents, `INSERT ... ON CONFLICT DO UPDATE`) →
  `15_build_curve.sql` → `20_views.sql` → `analytics/3x_*.sql`. Base reconstruite par
  `python -m energycurves.db` ; fichier `.duckdb` jamais committé.
- `sql/analytics/*.sql` : en-tête (question métier, entrées, sortie, piège évité), pas de `SELECT *`,
  dates en `>= début AND < fin`, fenêtres avec `PARTITION BY` et cadre explicites, filtre d'affichage
  après le calcul des fenêtres, jointures temporelles en ASOF (`>=`).
- Toute logique codée deux fois (Python et SQL) est testée à l'égalité.

## Conventions financières

- Unités : CL \$/bbl ; RB, HO \$/gal (×42 → \$/bbl) ; NG \$/MMBtu.
- $\tau$ en années, Act/365. Jours ouvrés NYMEX ≈ calendrier fédéral US + Good Friday, validé empiriquement.
- Roll yield $> 0 \iff$ backwardation.
- Rendements futures = rendements excédentaires ; Sharpe calculé directement dessus.

## Tests de cohérence financière (bloc spécifique)

Données et raccord
- Rendement à contrat constant : sur un panel synthétique à contrats connus, l'indice reconstitué
  égale le ratio de prix du contrat détenu.
- Rolls du calendrier = rolls détectés dans les données EIA pour ≥ 99 % des expirations (±1 jour).
- WTI 2020-04-20 : aucune propagation de NaN ni de $\ln$ d'un prix négatif.

M1
- $y > 0 \iff F^{(1)} > F^{(2)}$ ; courbe plate ⇒ $y = 0$ ; invariance de $y$ à un facteur d'échelle des prix.

M2
- Perturber les données après $t$ ne change pas les positions en $t$ (signal, désaisonnalisation, vol).
- XS : somme des poids nulle, exposition brute normalisée.
- P&L et Sharpe décroissants avec le coût.

M3
- Courbe plate, coûts > 0 ⇒ intrinsèque = 0 et aucune transaction.
- Cas à 2 mois, $V_0 = V_2 = 0$ : intrinsèque $= \min(\bar V, \bar I, \bar W)\,\big(D_2(F_2 - c_w) - D_1(F_1 + c_i)\big)^+$.
- Intrinsèque croissante en capacité, vitesse d'injection, vitesse de soutirage.
- Stock dans $[0, \bar V]$ à chaque date ; condition terminale respectée.
- LP = programmation dynamique par force brute sur une grille (3 mois).
- Prix × $c$ ⇒ valeur × $c$.
- Vol nulle ⇒ RI = intrinsèque ; sans coût de re-couverture, RI ≥ intrinsèque sur chaque chemin.
- Forwards simulés : moyenne MC = forward initial à 3 erreurs standard près (martingale).
- Calibration 2 facteurs : retrouve les paramètres sur données simulées (tolérance fixée).

M4
- $K = 0$ ⇒ Kirk = Margrabe exactement.
- Parité call-put du spread : $C - P = D(F_1 - F_2 - K)$.
- MC : erreur en $O(N^{-1/2})$ ; MC dans 3 erreurs standard de Margrabe pour $K = 0$.
- Calibration Schwartz : retrouve $(\kappa, \sigma)$ sur données simulées.

## Résultats clés (bloc spécifique)

1. Roll yields moyens par produit et part du temps en backwardation.
2. Carry TS / XS / long-only, brut vs désaisonnalisé : Sharpe, t-stat, MDD, sensibilité aux coûts.
3. Stockage : intrinsèque vs rolling intrinsic, part extrinsèque, sensibilités $(\kappa, \sigma_l, \rho)$.
4. (M4) Écart Kirk vs Monte Carlo, décomposé 2 jambes / panier.

## Limites à écrire noir sur blanc dans le README

- Échantillon futures EIA arrêté au 2024-04-05 (publication interrompue par l'EIA).
- Univers de 4 contrats énergie, dont 3 fortement corrélés : puissance statistique faible.
- Vol au-delà de 4 mois extrapolée par le modèle (aucune donnée publique de courbe longue en historique).
- RI = borne inférieure de la valeur du stockage ; granularité mensuelle ; pas de ratchets ni de vol saisonnière.
- Vols réalisées, pas implicites (aucune donnée d'options publique gratuite).
