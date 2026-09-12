# T1 — baseline'y na czystym benchmarku

Wygenerowane: 2026-09-12 10:33

| sieć | agent | yellow | ATT (10 ost. testów) | ATT (ostatni) | throughput | queue | n |
|---|---|---|---|---|---|---|---|
| cityflow_hangzhou4x4 | advancedmp (W1=1.0) | 0 | 311.3 | 311.3 | 2931 | 1.3 | 1 |
| cityflow_hangzhou4x4 | advancedmp (W1=1.0) | 5 | 321.0 | 321.0 | 2932 | 1.4 | 1 |
| cityflow_hangzhou4x4 | efficientmp | 0 | 311.2 | 311.2 | 2931 | 1.3 | 1 |
| cityflow_hangzhou4x4 | efficientmp | 5 | 327.9 | 327.9 | 2929 | 1.6 | 1 |
| cityflow_hangzhou4x4 | fixedtime | 0 | 517.3 | 517.3 | 2750 | 11.0 | 1 |
| cityflow_hangzhou4x4 | fixedtime | 5 | 575.6 | 575.6 | 2645 | 12.5 | 1 |
| cityflow_hangzhou4x4 | maxpressure | 0 | 347.7 | 347.7 | 2926 | 3.2 | 1 |
| cityflow_hangzhou4x4 | maxpressure | 5 | 360.8 | 360.8 | 2925 | 3.5 | 1 |
| cityflow_hangzhou4x4 | sotl | 0 | 352.8 | 352.8 | 2915 | 3.3 | 1 |
| cityflow_hangzhou4x4 | sotl | 5 | 375.6 | 375.6 | 2918 | 4.0 | 1 |
| cityflow_hangzhou4x4_4ph | advancedcolight | 0 | 306.6 | 306.5 ± 0.2 | 2930 | 1.0 | 3 |
| cityflow_hangzhou4x4_4ph | advancedcolight | 5 | 316.9 | 317.3 ± 0.5 | 2931 | 1.1 | 3 |
| cityflow_hangzhou4x4_4ph | advancedmp (W1=1.0) | 0 | 314.7 | 314.7 | 2930 | 1.5 | 1 |
| cityflow_hangzhou4x4_4ph | advancedmp (W1=1.0) | 5 | 325.4 | 325.4 | 2930 | 1.6 | 1 |
| cityflow_hangzhou4x4_4ph | colight | 0 | 326.0 | 325.6 ± 1.1 | 2928 | 2.1 | 3 |
| cityflow_hangzhou4x4_4ph | colight | 5 | 342.5 | 342.4 ± 2.3 | 2927 | 2.5 | 3 |
| cityflow_hangzhou4x4_4ph | efficientmp | 0 | 314.0 | 314.0 | 2932 | 1.4 | 1 |
| cityflow_hangzhou4x4_4ph | efficientmp | 5 | 330.1 | 330.1 | 2932 | 1.7 | 1 |
| cityflow_hangzhou4x4_4ph | fixedtime | 0 | 506.2 | 506.2 | 2748 | 10.5 | 1 |
| cityflow_hangzhou4x4_4ph | fixedtime | 5 | 568.9 | 568.9 | 2641 | 12.1 | 1 |
| cityflow_hangzhou4x4_4ph | gatppo | 5 | 317.2 | 317.5 ± 1.0 | 2930 | 1.1 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_meanpool] | 5 | 316.4 | 316.6 ± 0.3 | 2931 | 1.1 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_nognn] | 5 | 316.3 | 316.0 ± 0.2 | 2931 | 1.1 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_ep] | 5 | 329.2 | 329.0 ± 0.5 | 2930 | 1.6 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_pressure] | 5 | 341.1 | 329.3 ± 1.2 | 2931 | 1.7 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_queue] | 5 | 328.9 | 328.2 ± 0.5 | 2931 | 1.6 | 3 |
| cityflow_hangzhou4x4_4ph | maxpressure | 0 | 374.3 | 374.3 | 2916 | 4.7 | 1 |
| cityflow_hangzhou4x4_4ph | maxpressure | 5 | 392.7 | 392.7 | 2920 | 5.3 | 1 |
| cityflow_hangzhou4x4_4ph | presslight | 5 | 339.2 | 338.4 ± 0.5 | 2926 | 2.3 | 3 |
| cityflow_hangzhou4x4_4ph | sotl | 0 | 344.9 | 344.9 | 2924 | 2.9 | 1 |
| cityflow_hangzhou4x4_4ph | sotl | 5 | 367.2 | 367.2 | 2926 | 3.5 | 1 |
| cityflow_jinan3x4 | advancedcolight [sanity60] | 0 | 265.8 | 264.7 | 6186 | 5.0 | 1 |
| cityflow_jinan3x4 | advancedmp (W1=1.0) | 0 | 271.7 | 271.7 | 6180 | 6.0 | 1 |
| cityflow_jinan3x4 | advancedmp (W1=1.0) | 5 | 296.0 | 296.0 | 6162 | 8.0 | 1 |
| cityflow_jinan3x4 | efficientmp | 0 | 270.0 | 270.0 | 6186 | 5.7 | 1 |
| cityflow_jinan3x4 | efficientmp | 5 | 309.3 | 309.3 | 6162 | 9.4 | 1 |
| cityflow_jinan3x4 | fixedtime | 0 | 443.6 | 443.6 | 5939 | 27.9 | 1 |
| cityflow_jinan3x4 | fixedtime | 5 | 514.1 | 514.1 | 5768 | 34.9 | 1 |
| cityflow_jinan3x4 | maxpressure | 0 | 306.5 | 306.5 | 6163 | 11.2 | 1 |
| cityflow_jinan3x4 | maxpressure | 5 | 344.4 | 344.4 | 6135 | 15.3 | 1 |
| cityflow_jinan3x4 | sotl | 0 | 331.9 | 331.9 | 6155 | 14.1 | 1 |
| cityflow_jinan3x4 | sotl | 5 | 382.5 | 382.5 | 6129 | 20.2 | 1 |
| cityflow_jinan3x4_4ph | advancedcolight | 0 | 268.8 | 269.1 ± 0.5 | 6180 | 5.6 | 3 |
| cityflow_jinan3x4_4ph | advancedcolight | 5 | 293.5 | 294.7 ± 2.5 | 6173 | 7.9 | 3 |
| cityflow_jinan3x4_4ph | advancedmp (W1=1.0) | 0 | 274.3 | 274.3 | 6180 | 6.3 | 1 |
| cityflow_jinan3x4_4ph | advancedmp (W1=1.0) | 5 | 299.2 | 299.2 | 6174 | 8.6 | 1 |
| cityflow_jinan3x4_4ph | colight | 0 | 284.2 | 288.0 ± 10.4 | 6177 | 8.3 | 3 |
| cityflow_jinan3x4_4ph | colight | 5 | 323.6 | 328.5 ± 5.5 | 6157 | 12.6 | 3 |
| cityflow_jinan3x4_4ph | efficientmp | 0 | 272.2 | 272.2 | 6182 | 6.1 | 1 |
| cityflow_jinan3x4_4ph | efficientmp | 5 | 316.1 | 316.1 | 6151 | 10.5 | 1 |
| cityflow_jinan3x4_4ph | fixedtime | 0 | 424.7 | 424.7 | 5940 | 25.5 | 1 |
| cityflow_jinan3x4_4ph | fixedtime | 5 | 507.0 | 507.0 | 5740 | 33.8 | 1 |
| cityflow_jinan3x4_4ph | gatppo | 5 | 295.6 | 295.9 ± 0.6 | 6172 | 8.1 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_meanpool] | 5 | 291.7 | 291.4 ± 0.5 | 6174 | 7.4 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_nognn] | 5 | 291.8 | 291.2 ± 0.3 | 6174 | 7.4 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_ep] | 5 | 314.2 | 314.2 ± 1.0 | 6160 | 10.3 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_pressure] | 5 | 320.4 | 321.7 ± 1.2 | 6157 | 11.3 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_queue] | 5 | 313.3 | 316.5 ± 4.8 | 6144 | 10.7 | 3 |
| cityflow_jinan3x4_4ph | maxpressure | 0 | 316.8 | 316.8 | 6152 | 12.8 | 1 |
| cityflow_jinan3x4_4ph | maxpressure | 5 | 361.0 | 361.0 | 6133 | 17.8 | 1 |
| cityflow_jinan3x4_4ph | presslight | 5 | 321.3 | 319.2 ± 2.3 | 6163 | 11.5 | 3 |
| cityflow_jinan3x4_4ph | sotl | 0 | 318.7 | 318.7 | 6157 | 12.2 | 1 |
| cityflow_jinan3x4_4ph | sotl | 5 | 352.9 | 352.9 | 6139 | 15.8 | 1 |
