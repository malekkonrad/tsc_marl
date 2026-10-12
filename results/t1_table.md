# T1: baselines on the clean benchmark

Generated: 2026-10-11 16:51

| network | agent | yellow | ATT (last 10 tests) | ATT (last) | throughput | queue | n |
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
| cityflow_hangzhou4x4_4ph | advancedcolight | 0 | 306.6 | 306.5 +- 0.2 | 2930 | 1.0 | 3 |
| cityflow_hangzhou4x4_4ph | advancedcolight | 5 | 317.1 | 317.4 +- 0.5 | 2931 | 1.1 | 5 |
| cityflow_hangzhou4x4_4ph | advancedmp (W1=1.0) | 0 | 314.7 | 314.7 | 2930 | 1.5 | 1 |
| cityflow_hangzhou4x4_4ph | advancedmp (W1=1.0) | 5 | 325.4 | 325.4 | 2930 | 1.6 | 1 |
| cityflow_hangzhou4x4_4ph | colight | 0 | 326.0 | 325.6 +- 1.1 | 2928 | 2.1 | 3 |
| cityflow_hangzhou4x4_4ph | colight | 5 | 342.5 | 342.4 +- 2.3 | 2927 | 2.5 | 3 |
| cityflow_hangzhou4x4_4ph | efficientmp | 0 | 314.0 | 314.0 | 2932 | 1.4 | 1 |
| cityflow_hangzhou4x4_4ph | efficientmp | 5 | 330.1 | 330.1 | 2932 | 1.7 | 1 |
| cityflow_hangzhou4x4_4ph | fixedtime | 0 | 506.2 | 506.2 | 2748 | 10.5 | 1 |
| cityflow_hangzhou4x4_4ph | fixedtime | 5 | 568.9 | 568.9 | 2641 | 12.1 | 1 |
| cityflow_hangzhou4x4_4ph | gatppo | 5 | 317.2 | 317.5 +- 0.7 | 2929 | 1.1 | 5 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_meanpool] | 5 | 316.5 | 316.7 +- 0.3 | 2931 | 1.1 | 5 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_nognn] | 5 | 316.3 | 316.2 +- 0.3 | 2931 | 1.1 | 5 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_ep] | 5 | 329.2 | 329.0 +- 0.5 | 2930 | 1.6 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_pressure] | 5 | 341.1 | 329.3 +- 1.2 | 2931 | 1.7 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_abl_state_queue] | 5 | 328.9 | 328.2 +- 0.5 | 2931 | 1.6 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_fail] | 5 | 321.7 | 321.2 +- 0.3 | 2930 | 1.3 | 3 |
| cityflow_hangzhou4x4_4ph | gatppo [p1_uw_rule] | 5 | 321.1 | 321.6 +- 2.4 | 2931 | 1.4 | 3 |
| cityflow_hangzhou4x4_4ph | maxpressure | 0 | 374.3 | 374.3 | 2916 | 4.7 | 1 |
| cityflow_hangzhou4x4_4ph | maxpressure | 5 | 392.7 | 392.7 | 2920 | 5.3 | 1 |
| cityflow_hangzhou4x4_4ph | presslight | 5 | 339.2 | 338.4 +- 0.5 | 2926 | 2.3 | 3 |
| cityflow_hangzhou4x4_4ph | sotl | 0 | 344.9 | 344.9 | 2924 | 2.9 | 1 |
| cityflow_hangzhou4x4_4ph | sotl | 5 | 367.2 | 367.2 | 2926 | 3.5 | 1 |
| cityflow_hangzhou4x4_4ph_r5816 | efficientmp | 5 | 428.1 | 428.1 | 5280 | 10.6 | 1 |
| cityflow_hangzhou4x4_4ph_r5816 | fixedtime | 5 | 559.5 | 559.5 | 4379 | 18.2 | 1 |
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
| cityflow_jinan3x4_4ph | advancedcolight | 0 | 268.8 | 269.1 +- 0.5 | 6180 | 5.6 | 3 |
| cityflow_jinan3x4_4ph | advancedcolight | 5 | 293.5 | 294.4 +- 1.8 | 6173 | 7.9 | 5 |
| cityflow_jinan3x4_4ph | advancedmp (W1=1.0) | 0 | 274.3 | 274.3 | 6180 | 6.3 | 1 |
| cityflow_jinan3x4_4ph | advancedmp (W1=1.0) | 5 | 299.2 | 299.2 | 6174 | 8.6 | 1 |
| cityflow_jinan3x4_4ph | colight | 0 | 284.2 | 288.0 +- 10.4 | 6177 | 8.3 | 3 |
| cityflow_jinan3x4_4ph | colight | 5 | 323.6 | 328.5 +- 5.5 | 6157 | 12.6 | 3 |
| cityflow_jinan3x4_4ph | efficientmp | 0 | 272.2 | 272.2 | 6182 | 6.1 | 1 |
| cityflow_jinan3x4_4ph | efficientmp | 5 | 316.1 | 316.1 | 6151 | 10.5 | 1 |
| cityflow_jinan3x4_4ph | fixedtime | 0 | 424.7 | 424.7 | 5940 | 25.5 | 1 |
| cityflow_jinan3x4_4ph | fixedtime | 5 | 507.0 | 507.0 | 5740 | 33.8 | 1 |
| cityflow_jinan3x4_4ph | gatppo | 5 | 295.7 | 295.8 +- 1.2 | 6174 | 8.0 | 5 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_meanpool] | 5 | 291.7 | 291.3 +- 0.5 | 6176 | 7.4 | 5 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_nognn] | 5 | 291.9 | 291.7 +- 0.8 | 6175 | 7.5 | 5 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_ep] | 5 | 314.2 | 314.2 +- 1.0 | 6160 | 10.3 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_pressure] | 5 | 320.4 | 321.7 +- 1.2 | 6157 | 11.3 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_abl_state_queue] | 5 | 313.3 | 316.5 +- 4.8 | 6144 | 10.7 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_fail] | 5 | 303.1 | 302.0 +- 1.7 | 6170 | 8.9 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_uw] | 5 | 302.5 | 303.7 +- 1.1 | 6174 | 9.1 | 3 |
| cityflow_jinan3x4_4ph | gatppo [p1_uw_rule] | 5 | 301.7 | 299.9 +- 1.1 | 6169 | 8.6 | 3 |
| cityflow_jinan3x4_4ph | maxpressure | 0 | 316.8 | 316.8 | 6152 | 12.8 | 1 |
| cityflow_jinan3x4_4ph | maxpressure | 5 | 361.0 | 361.0 | 6133 | 17.8 | 1 |
| cityflow_jinan3x4_4ph | presslight | 5 | 321.3 | 319.2 +- 2.3 | 6163 | 11.5 | 3 |
| cityflow_jinan3x4_4ph | sotl | 0 | 318.7 | 318.7 | 6157 | 12.2 | 1 |
| cityflow_jinan3x4_4ph | sotl | 5 | 352.9 | 352.9 | 6139 | 15.8 | 1 |
| sumo_krakow3w_high | advancedmp (W1=1.0) | 5 | 466.6 | 466.6 | 3465 | 4.5 | 1 |
| sumo_krakow3w_high | fixedtime | 5 | 390.2 | 390.2 | 3941 | 4.8 | 1 |
| sumo_krakow3w_high | maxpressure | 5 | 394.5 | 394.5 | 3662 | 2.6 | 1 |
| sumo_krakow3w_low | advancedmp (W1=1.0) | 5 | 282.0 | 282.0 | 1660 | 0.5 | 1 |
| sumo_krakow3w_low | fixedtime | 5 | 331.4 | 331.4 | 1618 | 1.4 | 1 |
| sumo_krakow3w_low | maxpressure | 5 | 274.4 | 274.4 | 1656 | 0.3 | 1 |
| sumo_krakow3w_mid | advancedmp (W1=1.0) | 5 | 283.0 | 283.0 | 2755 | 0.8 | 1 |
| sumo_krakow3w_mid | fixedtime | 5 | 336.7 | 336.7 | 2716 | 2.4 | 1 |
| sumo_krakow3w_mid | gatppo | 5 | 339.3 | 351.6 +- 62.2 | 2860 | 2.6 | 3 |
| sumo_krakow3w_mid | gatppo [k1_fail] | 5 | 361.7 | 358.7 +- 17.2 | 3878 | 1.5 | 3 |
| sumo_krakow3w_mid | gatppo [k2_core] | 5 | 359.9 | 376.3 +- 46.1 | 3792 | 1.9 | 3 |
| sumo_krakow3w_mid | gatppo [k2_fail] | 5 | 361.7 | 358.7 +- 17.2 | 3878 | 1.5 | 3 |
| sumo_krakow3w_mid | maxpressure | 5 | 276.2 | 276.2 | 2756 | 0.6 | 1 |
