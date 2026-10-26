# Teleoperation Wi-Fi Experiments

Standalone workspace for new teleoperation (Isaac Sim + Unity) measurements over Wi-Fi.

## Structure

- docs: Roadmaps and methodology.
- config: Scenario and network configuration.
- scripts: Automation for setup, collection, parsing.
- logs/raw: Unmodified logs from all components.
- logs/processed: Parsed and aligned logs.
- data/raw: Packet captures and telemetry dumps.
- data/processed: Clean datasets for analysis.
- results/figures: Plots and visual outputs.
- results/tables: KPI and RCA summaries.
- metadata: Run manifests and experiment notes.

## Key docs

- docs/synchronized_collection_sop.md: Standard synchronized collection procedure across AP/operator/robot.
- docs/kpi_collection_guide.md: KPI definitions by layer.
- docs/experiment_campaign_guide.md: Scenario design and trial methodology.

## Run ID format

Use: YYYYMMDD_site_scenario_trial

Example: 20260407_lab_contention_t01
