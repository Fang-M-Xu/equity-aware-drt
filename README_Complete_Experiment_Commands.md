# Complete DRT Execution Commands

This document summarizes the commands used for formal reproduction of the current project and strictly separates two types of work:

1. **Project computation and experiments**: data processing, model training, scenario generation, vehicle prepositioning, travel matrices, ALNS, benchmarks, and raw OFAT experiments.
2. **Statistical analysis and paper figures**: read existing experimental results and generate statistical tests, summary tables, Chapter 5 figures, and spatial maps.


Run all commands from the project root in Windows PowerShell.

---

## 1. Environment Setup and Service Checks

```powershell
cd G:\HUJI-FMCDRT\FMC15_TravelAnalysis_V6

conda activate mayavi-env

python -m pip install -r .\requirements.txt
python -m pip install -r .\requirements_results.txt
python -m pip install -r .\requirements_maps.txt
python -m pip install -e .
```

Valhalla and OTP must be running before a formal experiment begins:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8002/status
Invoke-WebRequest -UseBasicParsing http://localhost:8080/otp/
```

Also confirm that `configs/base.yaml` contains:

```yaml
routing:
  allow_fallback_speed: false

otp:
  enabled: true
```

Formal experiments must not fall back to straight-line distance approximations when Valhalla fails.

## 2. Raw Data Splitting (First Run or Re-splitting Only)

Skip this step if the train, validation, and test sets already exist in `data/processed/experiment_data/`.

```powershell
python .\stage0_raw_to_splits\scripts\prepare_experiment_data.py `
  --config .\stage0_raw_to_splits\configs\data_pipeline.yaml
```

## 3. Input Checks, Road Travel Times, and Demand Panels

As preparation, copy the experimental data generated in Step 2 from `.\stage0_raw_to_splits\data\processed\experiment_data\` to `.\data\processed\experiment_data\`.

The files include:
`building_master.parquet`: the spatial base table containing each building and its coordinates, TAZ, `ai`, `gap_i`, POI, building use, and other attributes.

dwell_events_test.parquet
dwell_events_train.parquet
dwell_events_validation.parquet
These three files contain the original survey activity chains. Each row represents one person's stop at a location on a particular day, split into training, validation, and test sets.

survey_trips_test.parquet
survey_trips_train.parquet
survey_trips_validation.parquet
These three files contain all candidate origin–destination trips. Each trip is constructed from two consecutive dwell events and assigned to the training, validation, or test set.

```powershell
python .\scripts\00_validate_inputs.py `
  --config .\configs\base.yaml
Check that all inputs required by the formal experiment exist, that no household occurs across the train/validation/test splits, and that Valhalla is running correctly.

python .\scripts\04_enrich_survey_trip_times.py `
  --config .\configs\base.yaml
Route the split survey trips through the Valhalla network, calculate direct driving time and distance for every OD pair, infer the estimated departure time as “arrival time − direct driving time,” and save the routed trip data.
Output directory: `.\data\processed\model_inputs`
survey_trips_test_routed.parquet,
survey_trips_train_routed.parquet,
survey_trips_validation_routed.parquet

python .\scripts\05_build_zone_panel.py `
  --config .\configs\base.yaml
Convert individual survey trips into zone-level demand-prediction training data by aggregating trips into demand counts for each TAZ, day of week, and time window. This data is used by the subsequent LightGBM demand model.
Output directory: `.\data\processed\model_inputs`
zone_panel_train.parquet
zone_panel_validation.parquet
zone_panel_test.parquet

```

## 4. Formal Demand-Model Training and Evaluation

```powershell
python .\scripts\06_train_lightgbm.py `
  --config .\configs\base.yaml `
  --evaluate-test

Output files:
`models\lightgbm\zone_demand_model.joblib`: LightGBM model.
outputs\metrics:
`lightgbm_validation_predictions.parquet`: cell-level predictions; each row contains actual demand, the LightGBM prediction, and baseline predictions.
`lightgbm_test_predictions.parquet`: held-out test predictions with the same structure as the validation file.
`lightgbm_validation_method_comparison.csv`: validation metrics comparing LightGBM with every baseline.
`lightgbm_test_method_comparison.csv`: final held-out test comparison of all methods.
`lightgbm_all_method_comparison.csv`: combined validation and test results for direct use in paper tables and figures.
`lightgbm_feature_importance.csv`: LightGBM feature importance for paper outputs; `importance_gain` measures how much each variable reduced the loss.
`lightgbm_training_metadata.json`: training summary.
`lightgbm_validation_test_diagnostic.json`: checks whether the validation and test feature matrices are identical, guarding against accidental file duplication.

python .\scripts\07_train_transformer.py `
  --config .\configs\base.yaml `
  --evaluate-test

`activity_trip_transformer.pt`: Transformer model.
`vocabulary.json`: token mappings for zone, activity, departure, and duration, ensuring consistency across training, prediction, and result decoding.
`profile_schema.json`: personal and household profile features used by the Transformer and their column order, ensuring that request-generation inputs match training inputs exactly.
`transformer_training_metrics.json`: per-epoch training and validation loss, best epoch, and related records used to check convergence and reproducibility.
`transformer_validation_metrics.json`: validation performance, including accuracy, F1, Top-k, and duration MAE for destination zone, activity, and duration; mainly used for model selection and tuning.
`transformer_test_metrics.json`: final test-set performance and generalization on data excluded from training and tuning; use this preferentially in the paper's final report.
`transformer_validation_predictions.parquet`: trip-level validation labels, predictions, confidence, and Top-k predictions for error analysis.
`transformer_test_predictions.parquet`: trip-level test labels and predictions for traceability and plotting.
`transformer_all_method_comparison.csv`: comparison between the Transformer and all baselines, used to determine whether it outperforms simple historical patterns.
`transformer_class_distribution.csv`: class counts and proportions for destination zone, activity, and duration in the train/validation/test sets; used to assess class imbalance and interpret accuracy and macro-F1.
```

## 5. Generate Demand Scenarios and Waiting Candidates

```powershell
python .\scripts\08_generate_1030_scenarios.py `
  --config .\configs\base.yaml
```
### 5.1 Create Vehicle Waiting Candidates
```powershell
python .\scripts\09_build_waiting_candidates.py `
  --config .\configs\base.yaml

Output directory: `outputs\prepositioning`
`waiting_candidates.parquet`: vehicle waiting-location candidates on the road network.
`waiting_candidates.gpkg`: the same candidates in GIS format, used in ArcGIS to verify that candidates lie on the correct roads and within the correct TAZs.
`metrics_competitive\waiting_candidate_generation_summary.json`: generation summary containing the final candidate count, TAZ coverage, minimum/mean/maximum candidates per TAZ, and counts before and after road filtering and spatial deduplication.
```

## 6. Generate Results for Three Vehicle-Prepositioning Methods

The formal prepositioning methods currently supported by the code are `random`, `demand_only`, and `proposed`.

```powershell
python .\scripts\10_preposition_vehicles.py `
  --config .\configs\base.yaml `
  --method random

python .\scripts\10_preposition_vehicles.py `
  --config .\configs\base.yaml `
  --method demand_only

python .\scripts\10_preposition_vehicles.py `
  --config .\configs\base.yaml `
  --method proposed
```
`proposed` is a weighted combination of predicted demand, service gap, and vulnerability.

Output files:
`outputs\prepositioning`: prepositioning output directory.
`zone_to_candidate_time_sec.npy`: travel times from zones to candidate waiting locations.
`vehicle_initial_positions.parquet`: final vehicle locations selected by the prepositioning algorithm.

## 7. Build Travel-Time Matrices for the Three Prepositioning Methods

```powershell
python .\scripts\11_build_travel_matrices.py `
  --config .\configs\base.yaml `
  --preposition-method random

python .\scripts\11_build_travel_matrices.py `
  --config .\configs\base.yaml `
  --preposition-method demand_only

python .\scripts\11_build_travel_matrices.py `
  --config .\configs\base.yaml `
  --preposition-method proposed
```
Output files:
`outputs\operating_scenarios`: final scheduling-scenario directory.
`requests_all.parquet`: complete demand plus feasibility audit, identifying DRT trips that fail the basic constraints.
`requests_feasible.parquet`: feasible requests filtered from `requests_all.parquet`; this is the request set used by subsequent Greedy and ALNS scheduling.
`vehicles.parquet`: vehicle-prepositioning results.
`nodes.parquet`: node mapping for the travel-time matrix, including whether each node is a pickup or drop-off.
`travel_matrices.npz`: Valhalla road travel time from node i to node j, allowing ALNS to avoid repeated Valhalla calls.
`scenario_summary.json`: vehicle count, candidate count, feasible-request count, and related information for each scenario.

## 8. Run ALNS for One Scenario and Seed

This run is for comparison with public transit and walking. It is excluded from the operational evaluation and generates only `proposed_full` with relaxed vehicle resources.
```powershell
python .\scripts\12_run_alns.py `
  --config .\configs\base.yaml `
  --preposition-method proposed `
  --scenario-id 0 `
  --seed 2026 `
  --method-name proposed_full_35
```
`--method-name proposed_full_35` defines the method name used to distinguish outputs from different configurations. `35` indicates a fleet size of 35.

Input directory: `outputs\operating_scenarios\proposed\scenario_000\`
requests_all.parquet,vehicles.parquet,travel_matrices.npz,requests_feasible.parquet,nodes.parquet

Output file: `outputs\alns\proposed_full\scenario_000\seed_2026\vehicle_results.parquet`, containing the final schedule for each vehicle, including its route, stops, travel time, and related data.

## 8.5 Run ALNS in Batch for the Multimodal Comparison

These runs compare DRT with public transit and walking. They are excluded from the operational evaluation and generate only `proposed_full` with relaxed vehicle resources.
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_proposed_full_35.ps1
```
After completion, use this command to verify that all 30 scenarios finished:
```powershell
python -c "from pathlib import Path; root=Path(r'.\outputs\alns_competitive\proposed_full_35'); files=list(root.glob('scenario_*/seed_*/request_results.parquet')); print('completed scenarios =',len(files)); print(*[str(p) for p in files],sep='\n')"
```
Precheck the service rate:
```powershell
python .\scripts\12_check_proposed_full_35.py
```
Input data: `outputs\alns\proposed_full_35\scenario_*\request_results.parquet`
Output file: `outputs\metrics\proposed_full_35_precheck.csv`

## 9. Generate Walking and Public Transit Benchmarks

OTP must remain running during execution.

```powershell
python .\scripts\13_run_multimodal_benchmarks.py `
  --config .\configs\base.yaml
```
Output directory: `\outputs\benchmarks\`
`scenario_XXX.parquet`: multimodal benchmark results for every request in the scenario, including walking time, distance, walking accessibility within 15 minutes, transit availability, transit time, and transfer count. It is subsequently joined to `request_results.parquet` by `request_id` to compare DRT, walking, and public transit.

## 10. DRT Scheduling

First inspect the complete execution plan:

```powershell
python .\scripts\14_run_comparative_experiments.py `
  --config .\configs\base.yaml `
  --experiments .\configs\experiments_nature.yaml `
  --dry-run
```

After confirming the scenario count, method count, seed count, and output directory, run:

```powershell
python .\scripts\14_run_comparative_experiments.py `
  --config .\configs\base.yaml `
  --experiments .\configs\experiments_nature.yaml `
  --resume
```

If the local server has sufficient capacity, specify the worker count to run tasks in parallel, for example:

```powershell
python .\scripts\14_run_comparative_experiments.py `
  --config .\configs\base.yaml `
  --experiments .\configs\experiments_nature.yaml `
  --workers 4 `
  --resume
```
Mini test:
```powershell
python .\scripts\14_run_comparative_experiments.py `
  --config .\configs\base.yaml `
  --experiments .\configs\experiments_nature.yaml `
  --max-scenarios 2 `
  --max-solver-seeds 1
```

Output files:
```powershell
execution_plan.json: complete task plan for the comparative experiment, including scenarios, methods, seeds, and execution settings; used for reproduction and checking experiment scale.
run_status.csv: status of every task, including completed/skipped/failed state, duration, seed, and scenario; used to verify batch completion.
summary.json: overall summary of the Step 14 batch, including total, completed, skipped, and failed tasks, total runtime, and workers.

Files produced for every scenario and seed under each of the six methods:
routes.parquet: final stop sequence for every vehicle, including pickup/drop-off nodes and visit order; usable for typical-route maps.
vehicle_results.parquet: vehicle-level operating results, including requests served, driving time and distance, and empty-running time and distance; used to analyze operating efficiency.
request_results.parquet: final result for every request, including served status, pickup delay, ride time, door-to-door time, gap, and social attributes; the main input for equity, multimodal, and service-gap-closure analyses.
summary.json: summary of the current ALNS run, including total requests, eligible, served, unserved, and service rate; used when aggregating scenarios and seeds.
```
## 11. OFAT Sensitivity Experiments

First inspect the plan for a single scenario:

```powershell
python .\scripts\15_run_sensitivity_analysis.py `
  --config .\configs\base.yaml `
  --sensitivity .\configs\sensitivity_ofat.yaml `
  --scenario-id 0 `
  --preposition-method proposed `
  --dry-run
```



Run the complete OFAT analysis for six representative scenarios:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_sensitivity_ofat.ps1
```

Verify that each scenario has the expected 13 results:

```powershell
python -c "import pandas as pd; p=r'outputs/metrics/sensitivity_ofat/sensitivity_ofat_all_scenarios.csv'; d=pd.read_csv(p); print('shape=',d.shape); print(d.groupby('scenario_id').size()); print('baseline count:'); print(d.groupby('scenario_id')['is_baseline'].sum())"
```

Summary output:

```text
outputs/metrics/sensitivity_ofat/sensitivity_ofat_all_scenarios.csv
Aggregates the OFAT sensitivity results across all scenarios to analyze how changes in key model parameters affect DRT service rate, equity, objective value, and computational performance.
```
---

## 12. Final Statistical Evaluation

```powershell
python .\scripts\16_evaluate_all.py `
  --config .\configs\base.yaml
```

Generated files:

```text
outputs/metrics/scenario_level_metrics.csv: scenario-level method performance. Results from multiple solver seeds for the same method and scenario are averaged to obtain service rate, gap-closure rate, pickup delay, ride time, door-to-door time, and other scenario-level metrics. Mainly used to compare the overall operating performance of Proposed with the baselines.
outputs/metrics/group_equity_metrics.csv: group-equity analysis. Reports service rate, 15-minute accessibility, and travel-time metrics for service-gap, low-income, carless, older, disabled, and student groups, as well as sector, gender, and age groups. Used to evaluate whether social groups receive equitable DRT access.
outputs/statistics/paired_wilcoxon_tests.csv: statistical significance tests of method differences. Performs paired Wilcoxon signed-rank tests between Proposed and every baseline over matched scenarios and records mean and median differences, p-values, and Holm-adjusted p-values.
outputs/statistics/bootstrap_confidence_intervals.csv: confidence intervals for key metrics. Uses bootstrap resampling of scenario metrics for every method to calculate the mean and 95% confidence interval, representing uncertainty and stability.
```
Equity statistics include:

```text
is_gap
is_older
is_80plus
is_disabled
is_student
is_low_income
is_no_car
sector_group
gender_group
age_group
income_group
vehicle_group
```
