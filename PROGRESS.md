# PROGRESS — energy-curves

## 2026-10-12 — Phase 0 (plan) — terminée

- Rédigés : `PLAN.md`, `CLAUDE.md`, `PROGRESS.md`.
- Vérifié : séries futures EIA contrats 1–4 (WTI, Henry Hub, RBOB, heating oil) arrêtées au 2024-04-05 ;
  spots EIA actifs (WTI, Brent, NYH gasoline, NYH heating oil, NYH ULSD, Henry Hub).
- Décisions :
  - univers = 4 contrats (pas de futures Brent ni propane à l'EIA) ;
  - échantillon figé au 2024-04-05, versionné dans `data/sample/` ;
  - courbe NG 12 mois : snapshot yfinance (a) si le spike réussit, chaînage saisonnier EIA (b) toujours implémenté ;
  - dynamique de courbe à 2 facteurs pour le rolling intrinsic (1 facteur ⇒ extrinsèque ≈ 0 par construction) ;
  - tests intégrés à chaque module ; MVP ≈ 23,5 h ; M4 ≈ 7 h (v1.1).
- Ouvert :
  - nom définitif du repo et du package ;
  - URL exacte du xls futures pétrole ; règle d'expiration NG ; rupture de spécification HO en 2013 ;
  - disponibilité et format des tickers de contrats individuels yfinance (ex. `NGJ27.NYM`, à vérifier).

## Prochaine séance : S1 (≈ 2,5 h)

1. Créer le repo, l'arborescence de `PLAN.md` §2, `pyproject.toml`, `.gitignore` (`data/raw/`).
2. `scripts/fetch_data.py` : télécharger les xls EIA futures et spot + FRED `DTB3` ; écrire les parquet
   et `MANIFEST.json` (URL, date d'extraction UTC, sha256, plage, unité).
3. Spike yfinance, 30 min maximum : courbe NG complète à date ; historique de contrats expirés.
   Noter la décision (a)/(b) et l'extension post-2024 ici.
4. Mesurer la taille de `data/sample/` ; confirmer qu'elle reste versionnable.
