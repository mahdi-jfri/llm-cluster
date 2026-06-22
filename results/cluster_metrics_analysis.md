# Cluster Metrics Analysis

Source file: `results/clinc_cluster_k150_full_exactk_vllm_qwen3-8b.json`

Final metrics are shown first. Candidate metrics are shown in parentheses.

| Metric | Value | Analysis |
|---|---:|---|
| `n_items` | 4500 | All 4,500 examples were evaluated. |
| `n_clusters` | 150 `(881)` | Final output has 150 clusters, matching the 150 gold labels. Candidate output has many more clusters, so it is over-split. |
| `n_labels` | 150 | There are 150 true intent labels. |
| `purity` | 0.0604 `(0.2680)` | Very low. Only about 6% of final assignments count toward each cluster's majority label. Candidate purity is better but still weak. |
| `inverse_purity` | 0.5967 `(0.4540)` | Moderate. For each true label, the best matching final cluster captures about 59.7% of that label on average. Since each label has 30 examples, that is roughly 18 of 30. |
| `pairwise_precision` | 0.0092 `(0.0145)` | Extremely low. If two items are placed in the same cluster, there is under 1% chance in the final output that they share the same true label. |
| `pairwise_recall` | 0.4498 `(0.2888)` | Moderate-low. About 45% of same-label item pairs are recovered in the same final cluster. This is much higher than precision, meaning clusters are broad and mixed. |
| `pairwise_f1` | 0.0181 `(0.0276)` | Very low. The precision/recall balance is poor, mainly because precision is extremely low. |
| `rand_index` | 0.6851 `(0.8687)` | Looks decent, but this metric is misleading here because most item pairs are from different labels and different clusters, so true negatives dominate. |
| `adjusted_rand_index` | 0.0055 `(0.0155)` | Near zero. After correcting for chance, the clustering is barely better than random. |
| `mutual_information` | 0.5853 `(1.8903)` | Low information overlap between predicted clusters and true labels. Candidate clusters preserve more label information. |
| `normalized_mutual_information` | 0.1774 `(0.4390)` | Weak. Final clusters capture only limited structure from the gold labels. Candidate output is meaningfully better but still not strong. |
| `homogeneity` | 0.1168 `(0.3773)` | Very low. Final clusters contain many different labels instead of being label-specific. |
| `completeness` | 0.3682 `(0.5249)` | Low-to-moderate. Items from the same true label are somewhat grouped together, but not cleanly. |
| `v_measure` | 0.1774 `(0.4390)` | Weak combined homogeneity/completeness score. Final result is poor; candidate result is better but still imperfect. |

## Bottom Line

The final clustering has high label mixing. The biggest warning signs are:

- `purity = 0.0604`
- `pairwise_precision = 0.0092`
- `adjusted_rand_index = 0.0055`
- `homogeneity = 0.1168`

Candidate metrics are consistently better on information-based scores, especially `normalized_mutual_information`, `homogeneity`, `completeness`, and `v_measure`, but they are still weak overall.
