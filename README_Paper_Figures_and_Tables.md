
# Paper Chapter 5 Tables and Figures

## chapter 5.1 Demand-Model and Scenario Validation
### 1.Table 5.1.1 — Zone-level demand prediction performance

```powershell
python .\scripts\17_generate_table_5_1_1.py `
  --config .\configs\base.yaml
```
Input files:
```powershell
Held-out test data: lightgbm_test_predictions.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_1_model\Table_5_1_1_zone_level_demand_prediction_performance.csv
```
### Run the Following Command Before Generating Figure 5.1.1
```powershell
python .\scripts\17_5_validate_demand_scenarios.py `
  --analysis-config .\configs\results_analysis.yaml `
  --base-config .\configs\base.yaml `
  --output-dir .\outputs\paper_results\chapter5_1_model\demand_validation
```
Input files:
```powershell
Held-out test data: lightgbm_test_predictions.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_1_model\demand_validation\
zone_prediction_comparison.csv
paper_table_demand_model.csv
```

### 2.Figure 5.1.1 Held-out zone demand prediction and high-demand zone identification

```powershell
python .\scripts\17_generate_figure_5_1_1split.py `
  --figure 5.1.1 `
  --zone-name-file .\data\raw\spatial\Jerusalem_90_zones.shp `
  --zone-label-mode name
```
Input files:
```powershell
outputs\paper_results\chapter5_1_model\demand_validation\
zone_prediction_comparison.csv
paper_table_demand_model.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_1_model\
Figure_5_1_1a_Editorial_Demand_Calibration.png
Figure_5_1_1b_Editorial_High_Demand_Zones.png
```

### 3.Figure 5.1.2 — Spatial distribution of observed and predicted demand
```powershell
python .\scripts\17_generate_figure_5_1_2.py `
  --config .\configs\base.yaml
```
Input files:
```powershell
Held-out test data: lightgbm_test_predictions.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_1_model\Figure_5_1_2_spatial_demand_plot.png
```
### 4.Table 5.1.2 — Validation of synthetic travel requests
### Figure 5.1.3 — Distributional comparison between held-out survey and synthetic requests
```powershell
python .\scripts\17_generate_synthetic_request_validation.py `
  --config .\configs\base.yaml `
  --candidate-dir .\outputs\forecast_scenarios `
  --taz-lookup .\data\raw\Jerusalem_TAZ_Accessibility.xlsx
```
Input files:
```powershell
survey_trips_test_routed.parquet
outputs\forecast_scenarios\forecast_scenario_*.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_1_model\
Table_5_1_2_distributional_validation_of_synthetic_requests
Figure_5_1_3_representative_distributional_comparison.png
```

## Chapter 5.2 RQ1: Effects of Demand-Informed and Equity-Weighted Vehicle Prepositioning

### 1.Figure 5.2.1 — Spatial distribution of mean road travel time from request origins to the nearest initially prepositioned vehicle
```powershell
Before running, make sure Valhalla is available at: http://localhost:8002

python .\scripts\17_generate_figure_5_2_1.py `
  --config .\configs\base.yaml `
  --output-dir .\outputs\paper_results\chapter5_prepositioning `
  --methods random demand_only proposed `
  --minimum-zone-requests 5 `
  --dpi 600 `
  --show-vehicle-points
```
Input files:
```powershell
outputs\operating_scenarios\requests_all.parquet
outputs\prepositioning\vehicle_initial_positions.parquet
outputs\operating_scenarios\travel_matrices.npz
```
Output files:
```powershell
outputs\paper_results\chapter5_prepositioning\
Figure_5_2_1_nearest_vehicle_road_time.png
Figure_5_2_1_TAZ_summary.csv
Figure_5_2_1_paired_difference.csv
Figure_5_2_1_method_summary.csv
```

### 2.Table 5.2.1 — Performance by vehicle prepositioning strategy
```powershell
python .\scripts\17_generate_table_5_2_1.py `
  --config .\configs\base.yaml
```
Input files:
```powershell
outputs\metrics\scenario_level_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_prepositioning\Table_5_2_1_operational_performance_by_prepositioning.csv
```

### 3.Figure 5.2.2 — Operational effects of gap–vulnerability-weighted vehicle prepositioning
```powershell
python .\scripts\17_generate_figure_5_2_2.py `
  --config .\configs\base.yaml `
  --bootstrap-repetitions 5000 `
  --seed 2026 `
  --dpi 600
```
Input files:
```powershell
outputs\metrics\scenario_level_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_prepositioning\
Figure_5_2_2_operational_effects.png
Figure_5_2_2_operational_effects_data.csv
```

## Chapter 5.3 RQ2: Operational Performance of Equity-Aware Selective Routing
### 1.Table 5.3.1 — Greedy and ALNS routing performance
```powershell
python .\scripts\17_generate_table_5_3_1.py `
  --config .\configs\base.yaml
```
Input files:
```powershell
outputs\metrics\scenario_level_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_routing\Table_5_3_1_greedy_vs_alns_routing_performance.csv
```

### 2.Figure 5.3.1 — Greedy versus ALNS performance across scenarios
```powershell
python .\scripts\17_generate_figure_5_3_1.py `
  --config .\configs\base.yaml
```
Input files:
```powershell
outputs\metrics\scenario_level_metrics.csv
```

Output files:
```powershell
outputs\paper_results\chapter5_routing\
Figure_5_3_1_greedy_vs_alns_across_scenarios.png
Figure_5_3_1_greedy_vs_alns_across_scenarios_data.csv
```

### 3.Figure 5.3.2 — Operational–equity trade-offs across routing objectives
```powershell
python .\scripts\17_generate_figure_5_3_2.py `
  --config .\configs\base.yaml
```

Input files:
```powershell
outputs\metrics\scenario_level_metrics.csv
```

Output files:
```powershell
outputs\paper_results\chapter5_routing\
Figure_5_3_2_operational_equity_tradeoff.png
Figure_5_3_2_operational_equity_tradeoff_data.csv
```

### Run the Following Command Before Generating Table 5.3.2 and Figure 5.3.3
```powershell
python .\scripts\17_5_evaluate_routing_equity.py `
  --analysis-config .\configs\results_analysis.yaml `
  --base-config .\configs\base.yaml
```
Input files:
```powershell
outputs/alns/proposed_full/scenario_*/seed_*/request_results.parquet
```
Output files:
```powershell
outputs\paper_results\routing_equity\routing_scenario_metrics.csv
outputs\paper_results\routing_equity\group_service_metrics.csv
```
### 4.Table 5.3.2 Equity and disparity metrics under different ALNS route objective configurations
```powershell
python .\scripts\17_generate_table_5_3_2.py
```
Input files:
```powershell
outputs\paper_results\chapter5_routing\routing_scenario_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_routing\
Table_5_3_2_equity_disparity_metrics.csv
```
### 5.Figure 5.3.3 Service rates across different groups under the Proposed-full configuration
```powershell
python .\scripts\17_generate_figure_5_3_3.py `
  --bootstrap-repetitions 5000 `
  --seed 2026 `
  --dpi 600
```
Input files:
```powershell
outputs\paper_results\chapter5_routing\routing_scenario_metrics.csv
outputs\paper_results\chapter5_routing\group_service_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_routing\
Figure_5_3_3_group_service_rates.png
Figure_5_3_3_group_service_rates_data.csv
```
### 6.Figure 5.3.4 — DRT service rates across population groups-abundance-abundance
```powershell
python .\scripts\17_generate_figure_5_3_4.py `
  --input .\outputs\metrics\group_equity_metrics.csv
```
Input files:
```powershell
outputs\metrics\group_equity_metrics.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_routing\
Figure_5_3_4_group_service_rates.png
Figure_5_3_4_group_service_rates_data.csv
```

## Chapter 5.4 RQ3: Multimodal Accessibility and Service-Gap Closure
### 1.Table 5.4.1 — Fifteen-minute accessibility by travel mode
```powershell
python .\scripts\17_generate_table_5_4_1.py `
  --benchmark-dir .\outputs\benchmarks `
  --alns-dir .\outputs\alns `
  --output-dir .\outputs\paper_results\chapter5_multimodal
```
Input files:
```powershell
outputs/benchmarks/scenario_*.parquet
outputs/alns/proposed_full/scenario_*/seed_*/request_results.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal_35\
Table_5_4_1_fifteen_minute_accessibility_by_mode.csv
Table_5_4_1_scenario_level_accessibility.csv
multimodal_request_level_base.csv
```
### Run the Following Command Before Generating Figure 5.4.1
```powershell
python .\scripts\17_5_analyze_drt_vs_benchmarks.py `
  --config .\configs\base.yaml `
  --method-name proposed_full `
  --benchmark-dir .\outputs\benchmarks `
  --alns-dir .\outputs\alns `
  --output-dir .\outputs\metrics\multimodal_evaluation `
  --threshold-min 15 `
  --bootstrap-repetitions 5000 `
  --seed 2026
```
Input files:
```powershell
outputs\alns\scenario_*.parquet
```
Output files:
```powershell
outputs\metrics\multimodal_evaluation\
paper_table_overall_modes.csv
group_drt_vs_benchmarks.csv
```

### 2.Figure 5.4.1 15-minute accessibility of DRT, walking and GTFS fixed-route public transport
```powershell
python .\scripts\17_generate_figure_5_4_1.py
```
Input files:
```powershell
outputs\metrics\multimodal_evaluation\paper_table_overall_modes.csv
outputs\metrics\multimodal_evaluation\group_drt_vs_benchmarks.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal\Figure_5_4_1_multimodal_accessibility.png
```
### 3.Table 5.4.2 DRT accessibility conversion for walking- and transit-disadvantaged requests
```powershell
python .\scripts\17_generate_table_5_4_2.py
```
Input files:
```powershell
outputs\metrics\multimodal_evaluation\request_consensus_across_seeds.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal\Table_5_4_2_DRT_accessibility_conversion.csv
```
### 4.Figure 5.4.2 Spatial accessibility gains of DRT relative to GTFS fixed-route bus transit
```powershell
python .\scripts\17_generate_figure_5_4_2.py
```
Input files:
```powershell
outputs\metrics\multimodal_evaluation\request_consensus_across_seeds.csv
outputs\forecast_scenarios_competitive\forecast_scenario_*.parquet
data\raw\spatial\Jerusalem_90_zones.shp
data\processed\model_inputs\zone_static_features.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal\
Figure_5_4_2_spatial_accessibility_gains.png 
Figure_5_4_2_zone_summary.csv
Figure_5_4_2_representative_taz.csv
```

### 5.Table 5.4.3. Group-level within-15-minute accessibility under DRT, direct walking, and GTFS transit
```powershell
python .\scripts\17_generate_table_5_4_3.py
```
Input files:
```powershell
outputs\metrics\multimodal_evaluation\group_drt_vs_benchmarks.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal\Table_5_4_3_group_level_multimodal_accessibility.csv
```
### 6.Figure 5.4.3. Group within-15-minute accessibility under DRT, walking, and GTFS public transit
```powershell
python .\scripts\17_generate_figure_5_4_3.py `
  --input .\outputs\metrics\multimodal_evaluation\group_drt_vs_benchmarks.csv `
  --output-dir .\outputs\paper_results\chapter5_multimodal `
  --dpi 600
```
Input files:
```powershell
outputs\metrics\multimodal_evaluation\group_drt_vs_benchmarks.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_multimodal\
Figure_5_4_3_group_multimodal_accessibility.png
Figure_5_4_3_group_multimodal_plot_data.csv
```

## Chapter 5.5 Sensitivity, Robustness, and Computational Performance
### 1.Table 5.5.1 — Comparative ablation results for methodological components
```powershell
python .\scripts\17_generate_table_5_5_1.py `
  --config .\configs\base.yaml `
  --alns-dir .\outputs\alns `
  --output-dir .\outputs\paper_results\chapter5_sensitivity `
```
Input files:
```powershell
outputs/alns/proposed_full/scenario_*/seed_*/request_results.parquet
```
Output files:
```powershell
outputs\paper_results\chapter5_sensitivity\
    Table_5_5_1_comparative_ablation_results.csv
    Table_5_5_1_comparative_ablation_results.tex
```
### 2.Figure 5.5.1 — OFAT sensitivity of key performance indicators
```powershell
python .\scripts\17_generate_figure_5_5_1.py `
  --input .\outputs\metrics\sensitivity_ofat\sensitivity_ofat_all_scenarios.csv
```
Input files:
```powershell
outputs\metrics\sensitivity_ofat\
scenario_000.csv
scenario_005.csv
scenario_011.csv
scenario_017.csv
scenario_029.csv
sensitivity_ofat_all_scenarios.csv
```
Output files:
```powershell
outputs\paper_results\chapter5_sensitivity\
Figure_5_5_1_ofat_sensitivity_kpis.png
Figure_5_5_1_ofat_sensitivity_summary.csv
Figure_5_5_1_ofat_sensitivity_scenario_level.csv
```

### 3.Figure 5.5.2 — Robustness across demand scenarios
```powershell
python .\scripts\17_generate_figure_5_5_2.py `
  --input .\outputs\metrics\scenario_level_metrics.csv
```
Input files:
```powershell
scenario_level_metrics.csv
```

Output files:
```powershell
outputs\paper_results\chapter5_sensitivity\
Figure_5_5_2_robustness_across_demand_scenarios.png
Figure_5_5_2_scenario_distribution_data.csv
Figure_5_5_2_robustness_summary.csv
```
### 4.Table 5.5.2 — Computational performance
```powershell
python .\scripts\17_generate_table_5_5_2.py `
  --config .\configs\base.yaml `
  --alns-dir .\outputs\alns `
  --output-dir .\outputs\paper_results\chapter5_sensitivity
```
Input files:
```powershell
outputs\alns\method\scenario_*\seed_*\summary.json
```

Output files:
```powershell
outputs\paper_results\chapter5_sensitivity\
Table_5_5_2_computational_performance.csv
Table_5_5_2_computational_performance.tex
Table_5_5_2_computational_performance_audit.csv
```

