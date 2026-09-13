# Bramka 2.2 — próba poprawy na martwych detektorach

Kryterium ustalone PRZED uruchomieniem: wariant zalicza, jeśli AUROC na błędnych odczytach ≥ 0.7 na OBU niewidzianych maskach (mask0p5, mask0p25_intersection) i ≥ 0.8 na każdym niewidzianym szumie (gauss3p5, gauss4, urand3p5). Trening na clean/gauss1/gauss2/mask0p25, ewaluacja na ogonie epizodu.

| wariant | gauss1 | gauss2 | gauss4 | urand3p5 | mask0p25 | mask0p5 | mask0p25_intersection | drift0p3 | pokrycie 90% (czyste) | zalicza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| okno 4 (bramka 2.2) | 0.797 | 0.866 | 0.906 | 0.872 | 0.700 | 0.630 | 0.603 | 0.534 | 85.3% | NIE |
| okno 12 | 0.822 | 0.898 | 0.931 | 0.908 | 0.738 | 0.689 | 0.602 | 0.536 | 68.8% | NIE |
| okno 12 + trwałość | 0.856 | 0.903 | 0.922 | 0.913 | 0.788 | 0.739 | 0.651 | 0.536 | 66.7% | NIE |

## Werdykt

**Bramka 2.2 ZAMKNIĘTA JAKO ZALICZONA CZĘŚCIOWO.** Żaden wariant nie osiąga 0.7 na obu niewidzianych maskach. Najlepszy na maskach: „okno 12 + trwałość” (min AUROC 0.651). Głowa jest estymatorem niepewności dla awarii SZUMOWYCH, a na awariach CICHYCH (martwe detektory) działa słabo — do opisania wprost; ablacja rule-gate vs UQ na maskach zyskuje na wadze.
