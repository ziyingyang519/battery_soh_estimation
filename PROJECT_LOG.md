## Research Question

Can the state of health (SOH) of battery RW3 be predicted using only the first 5 seconds of each random-walk discharge segment?

This is an independent baseline project. It is inspired by the feature ideas in Jia et al., but it is not an exact reproduction of the ISSA-DELM paper.

## Dataset

- Source: NASA Randomized Battery Usage 2: Room Temperature Random Walk.
- Battery used in the first experiment: RW3.
- Raw file: `data/raw/RW3.mat`.
- Total operating steps: 12,826.
- Random-walk discharge steps: 5,463.
- Reference discharge steps: 44.
- Each random-walk discharge step is sampled every 1 second and lasts 300 seconds.
- Random-walk discharge current is approximately constant within each step, but varies across steps.

Useful sources:

- NASA data repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Jia et al. (2022): <https://doi.org/10.1016/j.isci.2022.103988>

## SOH Label Construction

1. Select each step whose `comment` is `reference discharge`.
2. Calculate reference capacity by Ah integration:

   `capacity_ah = integral(abs(current) dt) / 3600`

3. Define the initial reference capacity as the 100% SOH baseline:

   `Q0 = 2.008966 Ah`

4. Calculate SOH:

   `SOH = capacity_ah / Q0 * 100`

The 44 reference labels decrease from 100% to approximately 53%.

For each random-walk discharge segment, SOH is assigned by linearly interpolating between neighboring reference-SOH measurements in experiment time. This is a label-construction assumption and should be stated in any report.

## Data Split

The feature rows are kept in chronological order and split without shuffling:

- Train: first 60%, 3,246 samples, SOH 77.50% to 99.44%.
- Validation: middle 20%, 1,082 samples.
- Test: final 20%, 1,082 samples, SOH 52.78% to 68.15%.

This is a strict future-aging test. The test set contains lower SOH values that are absent from training data.

## Experiment 1: Simple 5-Second Statistics

Features used:

- Mean current and current standard deviation.
- Start voltage, voltage after 5 seconds, voltage drop, mean voltage, and voltage standard deviation.

Results on the chronological test set:

| Model | Test MAE | Test RMSE |
| --- | ---: | ---: |
| Linear SVR | 14.982% | 16.503% |

Observation:

- Linear SVR could produce values below the minimum training SOH, so it had better test performance.

## Experiment 2: Jia-Inspired IHIs Without Voltage Control

Features used:

- `IRWST = (V_start - V_start+5s) / I_start+5s`
- `VSDWST = standard deviation of voltage from 0 to 5 seconds`

These are based on the two indirect health indicators in Jia et al. (2022): internal resistance within a short time (IRWST) and voltage standard deviation within a short time (VSDWST).

Results on the chronological test set:

| Model | Test MAE | Test RMSE |
| --- | ---: | ---: |
| Linear SVR | 19.742% | 20.560% |

Observation:

- This feature setting failed to follow the true SOH decline.
- Applying IRWST and VSDWST to random discharge fragments with mixed starting voltages/SOC performed worse than the simple-statistics baseline.

Important interpretation:

Jia et al. do not simply pool every 5-second fragment with arbitrary starting voltage. Their method extracts IHIs at controlled starting voltages and averages them within 0.05 V voltage intervals for selected voltage windows. Therefore, Experiment 2 uses the paper's feature formulas but not its full voltage-window aggregation procedure.

## Current Conclusion

Using only a per-segment 5-second snapshot is not yet sufficient for reliable future-SOH prediction of RW3 under the current strict chronological split. The largest issues observed are:

- Test SOH is substantially lower than training SOH, requiring extrapolation.
- Starting voltage/SOC and random discharge current create large variation in short-time voltage response.
- The paper-derived IHIs require voltage-window control to be used as intended.

## Next Planned Experiment

Rebuild the feature table using only random-walk discharge segments with starting voltage between 4.0 V and 4.2 V.

Rationale:

- This matches one of the voltage windows evaluated by Jia et al.
- It reduces the SOC-related variation caused by mixing fragments that start at very different voltages.
- The features will remain IRWST and VSDWST within the first 5 seconds.

After rebuilding the table:

1. Record the new sample count.
2. Retrain Linear SVR using the same chronological split.
3. Compare MAE, RMSE, prediction ranges, and the test-set plot against Experiments 1 and 2.

## Project Files

- `src/inspect_rw3.py`: raw MATLAB-data inspection and initial SOH-label checks.
- `src/build_feature_table.py`: creates the 5-second feature table.
- `src/train_svr_baseline.py`: chronological split, Linear SVR training, metrics, and test plot.
- `data/processed/rw3_features_4p0_to_4p2_audit.csv`: current audited feature table.
- `outputs/svr_test_predictions_4p0_to_4p2_group_aware.png`: current test prediction plot.

## 實驗 2：固定起始電壓範圍

- 依據 Jia 論文的固定電壓區間概念，只保留 random-walk 放電開始電壓介於 4.0–4.2 V 的片段。
- 特徵仍為 5 秒的 IRWST 與 VSDWST。
- 可用樣本：428 筆；chronological split 為 Train 256 / Validation 86 / Test 86。

| Model | Train RMSE | Validation RMSE | Test RMSE |
|---|---:|---:|---:|
| Linear SVR | 3.355% | 3.896% | 4.221% |

結論：控制起始電壓後，Linear SVR 的測試誤差明顯改善，表示起始電壓／SOC 是影響短時間健康特徵的重要條件。

## 實驗 3：起始電壓範圍比較

固定條件：
- 資料：RW3 random-walk discharge
- 特徵：前 5 秒 IRWST、VSDWST
- 模型：Linear SVR
- 切分：chronological 60% / 20% / 20%

| Start-voltage window | Samples | Linear SVR test RMSE |
|---|---:|---:|
| 4.0–4.2 V | 428 | 4.221% |
| 3.8–4.0 V | 763 | 17.638% |
| 3.6–3.8 V | 1,430 | 21.171% |

結論：4.0–4.2 V 的起始電壓範圍在本實驗中效果最佳。選擇 Linear SVR 作為目前主模型，因為它在該範圍的 validation RMSE 為 3.896%、test RMSE 為 4.221%，且能對低 SOH 區間做較合理的外插。

## 實驗 4：Feature ablation

資料範圍固定為起始電壓 4.0–4.2 V，模型固定為 Linear SVR。

| Features | Validation RMSE | Test RMSE |
|---|---:|---:|
| IRWST only | 3.874% | 4.200% |
| VSDWST only | 11.427% | 19.549% |
| IRWST + VSDWST | 3.896% | 4.221% |

結論：IRWST 是主要健康特徵。VSDWST 單獨預測效果不佳，加入 IRWST 後也未帶來實質改善，因此主 baseline 選擇 IRWST only。

## 實驗 5：Linear Regression 與 Linear SVR

固定條件：
- RW3
- 起始電壓 4.0–4.2 V
- 特徵：IRWST only
- chronological split：60% / 20% / 20%

| Model | Validation RMSE | Test RMSE |
|---|---:|---:|
| Linear Regression | 4.035% | 4.248% |
| Linear SVR | 3.874% | 4.200% |

結論：IRWST 與 SOH 呈高度近似線性關係。Linear SVR 略優於 Linear Regression，故作為主 baseline；Linear Regression 保留為可解釋的簡單比較模型。

## 實驗 6：跨電池泛化測試

設定：
- 訓練資料：全部 RW3 samples
- 外部測試資料：全部 RW4 samples
- 起始電壓範圍：4.0–4.2 V
- 特徵：IRWST only
- RW4 SOH 僅用於最終 MAE、RMSE 評估，未參與訓練。

| Model | RW3 Train RMSE | RW4 External Test MAE | RW4 External Test RMSE |
|---|---:|---:|---:|
| Linear Regression | 3.608% | 2.846% | 4.251% |
| Linear SVR | 3.639% | 2.832% | 4.224% |

結論：RW3 訓練的 IRWST-only 模型可泛化到未見過的 RW4，Linear SVR 在 RW4 的 external-test RMSE 為 4.224%。但 Linear SVR 與 Linear Regression 的差異極小，顯示此設定下的 IRWST-SOH 關係高度近似線性。

限制：RW3 與 RW4 來自相同 NASA 資料集與相近操作條件；尚未驗證跨溫度、跨化學材料或跨資料集泛化。

## 實驗 7：物理限制與因果時間平滑

目的：降低單段 5 秒 IRWST 預測的尖峰誤差。

方法：
- Physical clipping：將預測 SOH 限制在 0–100%。
- Causal moving median：每筆只使用目前與前 14 筆 prediction；窗口大小 15 由 RW3 validation 選出，未使用 RW4 選擇。

| RW4 external test, Linear SVR | MAE | RMSE |
|---|---:|---:|
| Raw prediction | 2.832% | 4.224% |
| Clipped prediction | 2.735% | 4.004% |
| Clipped + causal median (15 samples) | 2.234% | 3.099% |

結論：clipping 可修正超出物理範圍的預測；causal median 可抑制單段 IRWST 的尖峰，使跨電池 RMSE 降至 3.099%。但此結果使用了歷史 prediction，因此應稱為 sequence-enhanced estimation，而非純單段 5 秒估測。

## 實驗 8：跨電池 IRWST-SOH 關係比較

設定：
- 電池：RW3、RW4、RW5、RW6
- 起始電壓範圍：4.0–4.2 V
- 特徵：IRWST only

| Battery | Pearson r | Linear slope |
|---|---:|---:|
| RW3 | -0.954 | -896.8 |
| RW4 | -0.921 | -888.2 |
| RW5 | -0.933 | -900.7 |
| RW6 | -0.797 | -1646.8 |

結論：
- RW3、RW4、RW5 的 IRWST-SOH slope 約為 -900，關係相近，因此 RW3 訓練的模型可泛化至 RW4、RW5。
- RW6 的 slope 為 -1646.8，約為 RW3 的 1.8 倍；表示相同 IRWST 變化在 RW6 對應更快的 SOH 下降。
- RW6 表現較差的主要原因是 cell-to-cell calibration difference，而非 IRWST 完全失效。
- 後續可研究使用少量目標電池早期資料進行 calibration 或 domain adaptation。

## 實驗 9：RW6 early-life affine calibration

設定：
- Base model：RW3 訓練的 Linear SVR，IRWST only。
- Calibration data：RW6 最早 10%（16 samples）。
- Final test：RW6 最後 20%（34 samples）。
- Calibration model：以 base prediction 預測 true SOH 的 Linear Regression。

結果：
- Calibration SOH range：96.60% - 98.98%。
- Learned equation：
  SOH_calibrated = -1.102 x SOH_base + 198.978
- RW6 final-test base model：MAE 11.512%，RMSE 11.579%。
- RW6 final-test early-life calibrated model：MAE 27.785%，RMSE 27.888%。

結論：
- 最早期資料的 SOH 範圍過窄，無法穩定估計 target cell 的 IRWST-SOH slope。
- 負 slope 與 100% clipping 顯示 affine calibration 發生不可靠的外插。
- 若要校正 RW6，需使用涵蓋較寬 SOH 範圍的 calibration data 或週期性 reference measurement。

## 實驗 10：RW6 two-point periodic calibration

設定：
- Base model：RW3 訓練的 Linear SVR，IRWST only。
- Early anchor：RW6 samples 1-10，平均 true SOH = 98.40%。
- Midlife anchor：RW6 samples 93-102，平均 true SOH = 83.23%。
- Final test：RW6 最後 20%（34 samples），SOH 範圍 67.79% - 76.57%。

結果：
- Calibration equation：
  SOH_calibrated = 2.394 x SOH_base - 120.931
- RW6 final-test base model：MAE 11.512%，RMSE 11.579%。
- RW6 final-test two-point calibrated model：MAE 7.261%，RMSE 7.600%。

結論：
- 一個早期與一個中期 anchor 可部分修正 RW6 的 cell-to-cell slope difference。
- 與 base model 相比，final-test RMSE 降低約 34.4%。
- 但 calibration anchors 最低僅到約 83% SOH，面對 68-77% SOH 的 final test 仍需外插；顯示更低 SOH 範圍的週期性 reference measurement 可能仍有必要。

## 實驗 11：RW6 calibration frequency study

固定條件：
- Base model：RW3 訓練的 Linear SVR，IRWST only。
- Physical clipping：0–100%。
- Causal moving median：15 samples。
- Final test：RW6 最後 20%（34 samples），SOH 範圍 67.79% - 76.57%。

Calibration anchors：
- Early：samples 1-10，true SOH mean = 98.40%。
- Midlife：samples 93-102，true SOH mean = 83.23%。
- Late：samples 119-128，true SOH mean = 80.00%。

| Calibration strategy | Final-test MAE | Final-test RMSE |
|---|---:|---:|
| 0 anchors (zero-shot) | 11.512% | 11.579% |
| 1 early anchor (offset only) | 18.279% | 18.321% |
| 2 anchors (affine) | 7.261% | 7.600% |
| 3 anchors (affine) | 4.515% | 4.918% |

結論：
- 單一 early anchor 無法校正 RW6 的 slope difference，且會使誤差惡化。
- 至少需要兩個不同 SOH 範圍的 anchors，才能同時校正 slope 與 offset。
- 三個週期性 anchors 可將 final-test RMSE 從 11.579% 降至 4.918%，但這是需要 target-cell reference information 的 calibrated setting，不可與 zero-shot 結果混為一談。

## 2026-09-13 Update: Data Audit and Measured-Reference Validation

### Audit and leakage checks

- Rebuilt the active RW3 table as `rw3_features_4p0_to_4p2_audit.csv`.
- Each feature row records five samples (`0 <= relativeTime < 5 s`), start/end
  voltage, end current, IRWST, neighboring reference anchors, and interpolation
  fraction.
- Added integrity checks for time order, five-sample windows, voltage selection,
  positive current, IRWST calculation, and interpolation bounds.
- A simple row-wise chronological split can place rows from one
  reference-to-reference interpolation interval in different splits. The active
  interpolation-label baseline therefore uses a group-aware split by
  `reference_interval_id`: train intervals 1-22, validation 23-30, test 31+.

### Raw reference-capacity check

Reference capacity and SOH were recomputed directly from Ah integration for all
four cells.

| Battery | Reference discharges | Initial capacity (Ah) | Final SOH |
|---|---:|---:|---:|
| RW3 | 44 | 2.0090 | 52.98% |
| RW4 | 38 | 1.9783 | 57.08% |
| RW5 | 42 | 1.9698 | 57.27% |
| RW6 | 22 | 1.9292 | 67.95% |

Plot: `outputs/reference_capacity_soh_rw3_to_rw6.png`.

### Experiment 12: Measured-reference validation of voltage windows

Purpose: compare predictions with capacity-SOH measured at unseen reference
discharges, rather than only with interpolated labels.

Method:

- For each target reference discharge, select the closest preceding eligible
  random-walk feature in the same reference interval.
- Use the capacity-SOH measured at that target reference discharge as the
  evaluation label.
- Compare the same six held-out reference discharges (orders 33, 35, 37, 39,
  41, and 43) across all voltage windows.
- The models in this experiment were still trained on interpolation labels;
  measured-reference SOH was used only for final evaluation.

| Start-voltage window | MAE | RMSE | Median time gap |
|---|---:|---:|---:|
| 3.6-3.8 V | 21.13% | 21.60% | 5.90 h |
| 3.8-4.0 V | 17.59% | 20.21% | 3.16 h |
| 4.0-4.2 V | **3.35%** | **3.85%** | 5.11 h |

Conclusion: the 4.0-4.2 V window is much more informative under the current
first-five-second IRWST plus Linear SVR setup. Its result cannot be explained
only by shorter time gaps because its median gap was not the smallest.

### Experiment 13: Measured-reference-only train/validation/test

Purpose: remove interpolation SOH from every stage of one RW3 workflow.

- Input: 4.0-4.2 V first-five-second IRWST.
- Label: capacity-SOH measured at the following reference discharge.
- One closest preceding feature was retained for each usable reference interval.
- This produced 21 directly measured feature-label pairs.
- Chronological split: 12 train / 4 validation / 5 final test samples.
- `C = 100` was selected by validation; the model was then refit on train plus
  validation before the final test.

| Split | MAE | RMSE |
|---|---:|---:|
| Train | 1.01% | 1.53% |
| Validation | 4.55% | 4.98% |
| Final test | 3.97% | 4.21% |

Limitation: the label is directly measured, but the selected feature and target
reference discharge are not simultaneous. Time gaps in the 21-pair dataset
range from about 2 to 58 hours.

### Experiment 14: Time-gap sensitivity analysis

The original chronological boundaries were held fixed so that gap rules did not
move the tested aging period. `C = 100` was fixed because the strict rule leaves
only two validation samples, making retuning unstable.

| Gap rule | Train / Validation / Test samples | Final-test RMSE |
|---|---:|---:|
| <= 6 h | 5 / 2 / 4 | 2.31% |
| <= 12 h | 9 / 2 / 5 | 4.67% |
| No limit | 12 / 4 / 5 | 4.21% |

Fair comparison on the four shared test reference discharges (35, 37, 39, 41):

| Gap rule | Shared-test MAE | Shared-test RMSE |
|---|---:|---:|
| <= 6 h | 2.21% | 2.31% |
| <= 12 h | **1.90%** | **2.12%** |
| No limit | 4.35% | 4.54% |

Current interpretation:

- Excluding long-gap pairs improves this preliminary RW3 measured-reference
  result.
- The strict 6-hour setting has too few samples for a strong performance claim.
- Measured-reference evaluation is more direct than interpolation-label
  evaluation, but does not provide continuous instantaneous SOH ground truth.
- The project now has an explicit trade-off: interpolation provides many proxy
  labels, while measured-reference provides few, more credible labels.

### New files

- `src/plot_reference_capacity_soh.py`
- `src/compare_rw3_measured_reference_windows.py`
- `src/evaluate_rw3_measured_reference_soh.py`
- `src/train_rw3_measured_reference_4p0_to_4p2.py`
- `src/evaluate_rw3_measured_reference_gap_thresholds.py`
- `data/processed/rw3_measured_reference_window_pairs.csv`
- `data/processed/rw3_4p0_to_4p2_measured_reference_split.csv`
- `data/processed/rw3_measured_reference_gap_threshold_summary.csv`
- `outputs/rw3_measured_reference_window_comparison.png`
- `outputs/rw3_4p0_to_4p2_measured_reference_split.png`
- `outputs/rw3_measured_reference_gap_threshold_sensitivity.png`

### Experiment 15: Time and reference-order sanity-check baselines

Question: does IRWST predict measured-reference SOH better than simple aging
progress information?

Definition of elapsed time:

- `elapsed_experiment_hours` is the total experimental time from the first
  reference discharge to the start of the current random-walk feature.
- It is not the duration of one random-walk segment. It includes accumulated
  random-walk operation, charging, resting, and reference tests.
- It is an aging-progress proxy: in this controlled dataset, later time usually
  means more accumulated use and lower SOH. It does not mean that SOH must
  decrease linearly with clock time in general.

Setup:

- Dataset: the same 21 RW3 4.0-4.2 V measured-reference pairs.
- Split: fixed chronological 12 train / 4 validation / 5 test pairs.
- All models were trained only on the 12 training pairs before the comparison
  test, so their test metrics are directly comparable.
- Reference order means the previous scheduled reference interval number. It is
  an optimistic experimental aging-progress proxy, not a feature assumed to be
  available in a real BMS.

| Model | Test RMSE |
|---|---:|
| Train-mean dummy | 25.44% |
| IRWST linear regression | 7.13% |
| IRWST Linear SVR | 5.35% |
| Elapsed-time linear regression | **2.15%** |
| Reference-order linear regression | 2.33% |

Conclusion:

- IRWST is substantially more useful than predicting only the training-SOH
  mean, so it contains health-related signal.
- However, elapsed experimental time predicted this RW3 test set more accurately
  than IRWST alone.
- Therefore, the current data do not yet show that IRWST provides more
  predictive information than simple aging progress.
- The earlier 4.21% IRWST result is not directly comparable to this table,
  because it retrained after using validation data; this sanity check uses only
  the training split for every model.

### Experiment 16: Cross-cell time baseline, RW3 to RW4

Purpose: test whether the time-versus-IRWST conclusion remains when the model is
trained on RW3 and evaluated on a different battery, RW4.

Setup:

- Direct measured-reference pairs only, using the closest 4.0-4.2 V feature
  before each target reference discharge.
- Predefined pairing rule: feature-to-reference gap `<= 12 h`.
- RW3 source data: 16 of 21 available pairs, SOH 52.75-96.74%.
- RW4 external test: 11 of 18 available pairs, SOH 57.52-95.11%.
- RW4 did not participate in model training.

| Model | RW4 external-test RMSE |
|---|---:|
| Train-mean dummy | 11.10% |
| IRWST linear regression | 2.36% |
| IRWST Linear SVR | 2.16% |
| Elapsed-time linear regression | **1.79%** |
| Reference-order linear regression | 1.92% |

Conclusion:

- The same pattern appears in the RW3-to-RW4 external test: elapsed time is
  slightly more accurate than IRWST alone.
- IRWST still transfers well and is far better than the dummy baseline, but it
  has not yet been shown to add predictive value beyond aging progress.
- This is still a small, same-dataset comparison (11 RW4 pairs), so it is
  evidence for a concern, not a final universal conclusion.

### Next implication

The next key test is whether a combined model using `elapsed time + IRWST`
improves on `elapsed time` alone. Only an improvement on the same held-out data
would support the claim that IRWST adds information beyond the age trend.

### Added files

- `src/compare_rw3_measured_reference_age_baselines.py`
- `src/compare_rw3_to_rw4_measured_reference_age_baselines.py`
- `data/processed/rw3_measured_reference_age_baseline_summary.csv`
- `data/processed/rw3_to_rw4_measured_reference_age_baseline_summary.csv`
- `outputs/rw3_measured_reference_age_baseline_comparison.png`
- `outputs/rw3_to_rw4_measured_reference_age_baseline_comparison.png`

### Experiment 17: Does IRWST add value beyond elapsed time across cells?

Purpose: test whether combining `elapsed time + IRWST` improves over either
feature alone when a model trained only on RW3 is transferred to RW4, RW5,
and RW6.

Setup:

- Direct measured-reference pairs only, with the same pre-defined rule as
  Experiment 16: closest 4.0-4.2 V random-walk feature before the target
  reference discharge and feature-to-reference gap `<= 12 h`.
- RW3 training set: 16 pairs.
- Three ordinary linear-regression models were trained on the same RW3 pairs:
  `elapsed time` only, `IRWST` only, and `elapsed time + IRWST`.
- RW4, RW5, and RW6 were used only as external test cells.

| Target cell | Test pairs | Time RMSE | IRWST RMSE | Time + IRWST RMSE |
|---|---:|---:|---:|---:|
| RW4 | 11 | 1.793% | 2.355% | **1.407%** |
| RW5 | 16 | 1.938% | 3.654% | **1.844%** |
| RW6 | 7 | **2.314%** | 9.921% | 3.202% |

Conclusion:

- For RW4 and RW5, adding IRWST to elapsed time lowered RMSE, but the RW5
  improvement was very small (1.938% to 1.844%).
- For RW6, elapsed time alone was clearly best; IRWST performed poorly and
  made the combined model worse.
- Therefore, this project currently cannot claim that IRWST consistently adds
  useful information beyond aging progress across cells. Its transferability
  appears cell-dependent.
- The evidence remains preliminary because all cells follow one laboratory
  protocol and RW6 has only seven eligible measured-reference test pairs.

### Added files (Experiment 17)

- `src/evaluate_rw3_to_rw4_rw5_time_plus_irwst.py`
- `data/processed/rw3_to_rw4_rw5_rw6_time_plus_irwst_summary.csv`
- `outputs/rw3_to_rw4_rw5_rw6_time_plus_irwst_comparison.png`

### Experiment 18: Updated main baseline using measured-reference SOH

Purpose: replace the earlier interpolation-label pipeline as the main model
evaluation baseline with a direct measured-reference formulation.

Main baseline definition:

- Battery: RW3.
- Input: IRWST from the first 5 s of a random-walk discharge that starts at
  4.0-4.2 V.
- Label: capacity-based SOH measured by the following reference discharge.
- Pairing: use only the closest preceding eligible feature for each reference
  discharge and require feature-to-reference gap `<= 12 h`.
- Split: chronological 60/20/20 by reference order: 9 train, 3 validation,
  and 4 final-test pairs.
- Model: Linear SVR, with `C` selected from validation candidates and
  `epsilon=0.1`.

Model selection and final test:

| Model | Validation RMSE | Final-test RMSE |
|---|---:|---:|
| Train-mean dummy | 18.534% | 28.787% |
| IRWST linear regression | 5.867% | 9.732% |
| IRWST Linear SVR, selected `C=10` | **2.600%** | **3.505%** |

Decision:

- The current main baseline model is **Linear SVR + IRWST**.
- Linear regression remains an interpretability comparator, and the dummy
  predictor remains the minimum reference point.
- `src/train_svr_baseline.py` remains an interpolation-label audit pipeline;
  it is no longer the project’s main measured-SOH performance baseline.

Limitation:

- The validation set contains only three pairs and the final test contains four
  pairs. The selected model is a defensible current starting point, not a final
  performance claim. It must be checked further near the 80% EOL threshold and
  across independent cells.

### Added files (Experiment 18)

- `src/train_measured_reference_baseline.py`
- `data/processed/rw3_measured_reference_main_baseline_pairs.csv`
- `data/processed/rw3_measured_reference_main_baseline_summary.csv`
- `outputs/rw3_measured_reference_main_baseline.png`

### Experiment 19: Partial correlation — does IRWST carry information beyond elapsed time?

Purpose: determine whether IRWST predicts SoH independently of elapsed
experimental time, or whether it is merely a proxy for aging progress.

Method:

- Dataset: all 16 RW3 measured-reference pairs (exploratory, no train/test
  split applied — uses full sample for diagnostic purposes).
- Three pairwise Pearson correlations computed: r(IRWST, SoH),
  r(elapsed_time, SoH), r(IRWST, elapsed_time).
- Partial correlation r(IRWST, SoH | elapsed_time): controls for elapsed time
  in both variables using the standard formula
  `r_partial = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))`,
  t-tested with df = n − 3.
- Residual model: regress SoH on elapsed time (OLS), correlate residuals with
  IRWST. This is an alternative collinearity-robust check.

Results:

| Quantity | Value | p-value |
|---|---:|---:|
| r(IRWST, SoH) | −0.978 | < 0.001 |
| r(elapsed_time, SoH) | −0.993 | < 0.001 |
| r(IRWST, elapsed_time) | +0.968 | < 0.001 |
| Partial r(IRWST, SoH \| elapsed_time) | −0.566 | 0.028 |
| r(IRWST, SoH residuals) | −0.142 | 0.601 |

Interpretation:

- The two methods give contradictory answers. The partial correlation (p =
  0.028) suggests IRWST carries an independent signal; the residual
  correlation (p = 0.60) does not.
- The contradiction is explained by extreme multicollinearity:
  r(IRWST, elapsed_time) = 0.968. IRWST and elapsed time are nearly the same
  variable in single-cell aging data. Because elapsed time already explains
  ~98.6% of SoH variance, the residuals are dominated by noise. IRWST cannot
  explain noise even if it explains health — hence residual r ≈ 0.
- Partial correlation controls for z in both directions and gives a different
  (more optimistic) answer; with n = 16 and extreme collinearity, both
  estimates are numerically sensitive.
- Conclusion: **statistically inconclusive with n = 16 and a single cell**.
  Neither method is fully reliable under these conditions.

Practical implication:

- Even if IRWST is only a proxy for aging-induced resistance growth (and thus
  collinear with time), it remains deployable in real BMS applications where
  calendar time may be unavailable or unreliable (e.g., reset counters,
  battery swaps). IRWST can be measured directly from a short discharge pulse.

### Added files (Experiment 19)

- `src/exp19_irwst_partial_correlation.py`
- `data/processed/exp19_irwst_partial_correlation_summary.csv`
- `outputs/exp19_irwst_partial_correlation.png`

### Experiment 20: Cross-cell partial correlation — pooled RW3+RW4+RW5+RW6

Purpose: increase sample size and reduce single-cell collinearity confound by
pooling all four cells (n = 64) for the partial correlation analysis.

Pairs per cell: RW3 = 16, RW4 = 18, RW5 = 20, RW6 = 10. Elapsed time was
not normalized across cells (raw experiment_time_s used).

Results:

| Quantity | Value | p-value |
|---|---:|---:|
| r(IRWST, SoH) | −0.926 | < 0.001 |
| r(elapsed_time, SoH) | −0.989 | < 0.001 |
| r(IRWST, elapsed_time) | +0.920 | < 0.001 |
| Partial r(IRWST, SoH \| elapsed_time) | −0.285 | 0.024 |
| r(IRWST, SoH residuals) | −0.112 | 0.379 |

Interpretation: The same pattern as Experiment 19 holds at larger n.
Partial r is significant (p = 0.024) but residual r is not (p = 0.38).
Multicollinearity decreased slightly (0.968 → 0.920) because different cells
have different IRWST–time trajectories, but remains too high for the residual
method to detect a weak independent signal. Conclusion: statistically
inconclusive. A cell-level normalization of elapsed time would be required
for a cleaner test.

### Added files (Experiment 20)

- `src/exp20_cross_cell_partial_correlation.py`
- `data/processed/exp20_cross_cell_partial_correlation_summary.csv`
- `outputs/exp20_cross_cell_partial_correlation.png`

### Experiment 21: RW6 failure diagnosis

Purpose: identify why IRWST-based SoH prediction fails for RW6 (RMSE ≈ 9.9%)
while RW4 and RW5 transfer acceptably (RMSE 2.4–3.7%).

Findings (three root causes):

**1. Abnormally slow resistance growth**

| Cell | IRWST growth rate |
|---|---:|
| RW3 | 11.97 mΩ / kh |
| RW5 | 10.58 mΩ / kh |
| RW4 | 8.84 mΩ / kh |
| RW6 | 4.37 mΩ / kh |

RW6's internal resistance grows at only one-third the rate of RW3. Its IRWST
range spans only 8–24 mΩ over ~2250 h, versus 10–60 mΩ for RW3 over ~3700 h.
IRWST is a far less sensitive health indicator for this cell.

**2. Non-monotonic IRWST trajectory**

RW6's IRWST drops from ~14 mΩ (SoH ≈ 88%) back to ~8 mΩ (SoH ≈ 85–80%)
and then rises again. Internal resistance should increase monotonically with
degradation. This reversal destroys the linear IRWST–SoH relationship within
RW6 (r drops to −0.66 vs −0.95 to −0.98 for other cells). Possible causes:
measurement condition change (temperature, current profile) around hours
1100–1600, or a short-lived cell-level recovery event.

**3. Steeper IRWST–SoH slope**

RW6's fitted slope (−1213) is 37% steeper than RW3 (−887). A model trained
on RW3 systematically over-estimates RW6's SoH. All RW6 prediction errors
are positive, confirming systematic over-estimation.

Conclusion: RW6 is not a typical difficult-transfer case — it is an anomalous
cell with fundamentally different resistance growth kinetics and a mid-life
measurement irregularity. Its failure mode is distinct from RW4/RW5 and
cannot be resolved by simple linear recalibration without addressing the
non-monotonic segment. IRWST-based SoH prediction requires per-cell
calibration. In a real BMS deployment, an initial reference discharge per
cell can establish cell-specific parameters. RW6 additionally suggests that
protocol anomalies or temperature variations can corrupt the IRWST signal.

### Added files (Experiment 21)

- `outputs/exp21_rw6_diagnosis.png`
