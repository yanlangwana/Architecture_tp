# Fiche modèle — `fraudlake-scoring` v1

| | |
|---|---|
| **Statut** | @champion (production) — 2026-11-21 |
| **Algorithme** | Gradient-Boosted Trees (`gbt_v2`), Spark MLlib |
| **Usage prévu** | Prioriser les transactions CB à investiguer / bloquer. Score = rang de risque (non calibré). |
| **Hors périmètre** | Décision automatique sans revue humaine ; autres pays / moyens de paiement. |
| **Données** | `/lakehouse/silver/transactions_enriched` @ version Delta 3 — train 2019-01 → 2020-04 (légitimes sous-échantillonnés à 10%), validation 2020-05 → 2020-06, test 2020-06 → 2020-12 |
| **Features** | 15 : amt_log, hour, is_night, is_weekend, is_online, age, distance_km, city_pop_log, txn_count_24h, amt_sum_24h, secs_since_prev_txn, amt_ratio_to_card_mean, days_to_holiday, category, gender |
| **Seuil** | 0.88 (F1 maximal sur la validation) |

## Performance (test, transactions jamais vues)
| PR-AUC | ROC-AUC | Précision | Rappel | Montant fraudé intercepté | Taux d'alerte |
|---|---|---|---|---|---|
| 0.926 | 0.9972 | 88.9% | 85.5% | 91.1% | 0.37% |

## Portes de qualité
| Porte | Valeur | Exigence | Résultat |
|---|---|---|---|
| PR-AUC test | 0.926 | ≥ 0.75 | ✅ validée |
| Rappel au seuil | 0.855 | ≥ 0.70 | ✅ validée |
| Précision au seuil | 0.889 | ≥ 0.50 | ✅ validée |
| Équité : rappel min / max (règle des 4/5) | 0.872 | ≥ 0.80 | ✅ validée |

## Équité (rappel par groupe)
| Dimension | Groupe | Fraudes | Rappel | Précision |
|---|---|---|---|---|
| genre | F | 1,164 | 81.4% | 86.9% |
| genre | M | 981 | 90.3% | 91.1% |
| tranche d'âge | 30–49 ans | 849 | 79.2% | 83.1% |
| tranche d'âge | 50–69 ans | 685 | 90.4% | 93.2% |
| tranche d'âge | 70 ans et + | 294 | 90.8% | 96.0% |
| tranche d'âge | < 30 ans | 317 | 87.1% | 88.2% |

## Limites & risques
- Données **synthétiques** (simulateur Sparkov) : motifs plus nets qu'en réalité, à revalider sur données réelles.
- Entraînement sous-échantillonné : recalibrer (Platt / isotonique) avant d'interpréter le score comme une probabilité.
- Dérive attendue (nouveaux modes opératoires, inflation) : surveillance PSI mensuelle et ré-entraînement si PSI ≥ 0.25.
- Équité : ratio de rappel min/max = 0.87 (écart 11.7%). Retirer `gender` ne réduit pas l'écart (testé : `gbt_v2_sans_genre`) — il provient de profils de fraude différents selon l'âge et le genre ; à surveiller à chaque version.
