# PROGRESS — commodities-curves-and-carry (package `energycurves`)

## État au 2026-11-05 — v1.0 livrée (modules 1 à 4)

Tout le plan est exécuté, module 4 compris (prévu en v1.1, finalement livré dans v1.0).
CI : ruff, mypy `--strict`, pytest + couverture, contrôle des hash de l'échantillon,
`run_results.py --check` (chiffres du README = code) et exécution du notebook.

Résultats clés (générés, voir README) :
- Courbes 2006–2024 : NG et WTI majoritairement en contango, RBOB majoritairement en
  backwardation ; une grande part de la pente NG/RB est saisonnière.
- SQL : corrélation négative, pour les 4 contrats, entre pente y14 et écart des stocks à leur
  moyenne 5 ans (théorie du stockage).
- Carry : aucune variante significative (|t| < 1) ; le meilleur Sharpe (XS désaisonnalisé)
  disparaît vers 10 bp de coûts et change de signe selon la désaisonnalisation.
- Stockage avr. 2027 – mars 2028 sur le strip du 2026-11-04 : intrinsèque 1.225, rolling
  intrinsic 1.271 ± 0.0007 USD/MMBtu (+3.8 %) ; extrinsèque porté par σ_s et κ seuls ; modèle à
  1 facteur calibré sur les mêmes données : +1.5 % ; courbe saisonnière moyenne : +35 %.
- Kirk vs Monte Carlo : erreur ATM ≤ 0.26 % ; jusqu'à +12 % très hors de la monnaie sur le
  3-2-1 (approximation du panier).

## 2026-11-05 — Séances S6 à S11 + module 4 (cette session)

Commits poussés sur `main` :
- `985038f` feat(storage): storage contract, forward curves and intrinsic value (module 3)
- `c70485d` feat(storage): two-factor curve model, calibration and exact simulation
- `46f90df` feat(storage): rolling-intrinsic valuation and parameter sensitivity
- `df9c43f` feat(spreads): crack-spread options, Kirk vs Monte Carlo (module 4)
- `1a71361` feat(results): one script regenerates every table, figure and document number
- `a7c40fd` feat(results): calendar check and one-factor storage benchmark
- `aa1de9a` docs: README, data notes and demo notebook
- `870b4cf` fix: read and write text files explicitly as UTF-8
- `3a8ea9f` ci: move checkout and setup-python to their Node 24 majors
- `44304c9` docs: progress log for sessions S1-S11, release notes and LinkedIn draft
- `e831ed8` docs: state the Kirk ATM result and the replay test precisely
- Tag `v1.0` créé localement mais **refusé au push (HTTP 403 du proxy de la session)** : à créer
  depuis GitHub (voir « Reste à faire »).

Décisions et écarts au plan :
- Valeur intrinsèque calculée deux fois : LP HiGHS et programmation dynamique sur grille
  vectorisée (exacte car matrice d'intervalles totalement unimodulaire) ; égalité testée,
  plus énumération exhaustive sur 3 mois. La DP rend le Monte Carlo du rolling intrinsic
  rapide (10 000 chemins < 1 s).
- Courbe NG : snapshot Yahoo (a) utilisé ; courbe saisonnière chaînée (b) publiée en
  robustesse. Pas d'extension des futures après 2024 via yfinance (inutile pour les résultats).
- Calibration 2 facteurs : les vols sont bien ajustées mais la corrélation C1–C2 est
  surestimée (0.99 modèle vs 0.95 données) → extrinsèque probablement conservateur (écrit
  dans le README).
- Ligne « 1 facteur (Schwartz 1997) » ajoutée au tableau de stockage pour justifier le choix
  du 2 facteurs avec des chiffres plutôt qu'un argument.
- Module 4 : calibration exacte (moyenne de σ² e^{−2κτ} sur les jours) au lieu d'une
  régression log-linéaire ; MC sans discrétisation sous le même modèle que Kirk.
- `run_results.py` réécrit tous les chiffres du README et de `docs/DATA.md` entre marqueurs ;
  aucun chiffre saisi à la main (le 98.7 % du calendrier est désormais calculé).
- `docs/DATA.md` n'avait pas été écrit en S1 contrairement à ce qu'indiquait la liste des
  tâches : rédigé dans cette session.
- Incident : le commit `1a71361` a échoué en CI (le script lisait `docs/DATA.md`, ajouté au
  commit suivant). Corrigé par `aa1de9a` ; aucun autre commit rouge.

## 2026-11-05 — Séances S1 à S5 + couche SQL (début de session)

Commits : `dfd4aad` plan, `1fd9c9d` squelette/CI, `2468d4d` données + MANIFEST, `c64abf6`
calendrier + rendements à contrat constant, `51c29ae` couche DuckDB (brief SQL), `a7503be`
module 1, `6388ee3` module 2.
- Push direct sur `main` (instructions du projet Claude, qui priment sur le brief SQL).
- Calendrier d'expiration validé sur les rolls identifiables par les prix (exceptions NG des
  années 1990 et 2008–2010 trouvées dans les données).

## 2026-10-12 — Phase 0 (plan) — terminée

- Rédigés : `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`.
- Vérifié : séries futures EIA contrats 1–4 arrêtées au 2024-04-05 ; spots EIA actifs.
- Décisions : univers de 4 contrats ; échantillon figé ; dynamique de courbe à 2 facteurs ;
  tests intégrés à chaque module.

## Reste à faire (actions de Tancrède)

1. Relire le README sur GitHub (rendu des formules et des tableaux).
2. Épingler le repo sur le profil GitHub (réglage de profil, non fait ici).
3. Créer la release `v1.0` sur GitHub (Releases → Draft a new release → nouveau tag `v1.0` sur
   `main`), avec les notes de version ci-dessous : le push du tag depuis la session a été
   refusé (403).
4. Publier le post LinkedIn (brouillon ci-dessous).

Pistes après v1.0 (non planifiées) : granularité journalière et ratchets pour le stockage,
volatilité saisonnière, vols implicites si une source gratuite apparaît, test du carry sur un
univers plus large (métaux, agricoles) pour gagner en puissance statistique.

## Notes de version v1.0

First complete release: NYMEX expiry calendar and constant-contract returns on public EIA data;
term-structure analytics; carry backtests (time series, cross section, long only) with
seasonal adjustment, costs and robustness; DuckDB layer with idempotent loads, data-quality
tests and documented analytics queries; natural-gas storage valuation (intrinsic by LP and
exact DP, rolling intrinsic under a calibrated two-factor curve model); crack-spread options
(Kirk and basket proxy vs exact Monte Carlo). Every README number is regenerated by
`scripts/run_results.py`, checked in CI.

## Brouillon LinkedIn (5 lignes)

1. What does an energy futures curve pay you to hold it, store it or trade its spreads?
2. I rebuilt 44 years of NYMEX expiries from public EIA data, added a DuckDB SQL layer, valued a gas storage contract (linear programme + rolling intrinsic under a calibrated two-factor curve model) and priced crack-spread options with Kirk vs exact Monte Carlo.
3. On the Nov-2026 Henry Hub strip, an Apr-27/Mar-28 storage contract is worth 1.225 USD/MMBtu intrinsic and 1.271 with monthly re-optimisation (+3.8%); carry on four energy contracts gives a Sharpe of 0.25 (t = 0.99), gone after costs.
4. Main lesson: much of the "carry" in gas and gasoline is seasonality, and storage optionality comes only from the short-term factor of the curve.
5. Code, tests and one-command reproduction: https://github.com/tancredeg/Commodities-curves-and-carry
