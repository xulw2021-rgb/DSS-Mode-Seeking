# Frozen publication results

These files are the corrected, frozen results used by the manuscript tables and
figures. They are regenerated only by rerunning the experiments; the
publication-artifact scripts read them directly.

- E1_learning_corrected.csv   CIGRE-14 main comparison, learning models (200 rows)
- E1_solvers.csv              CIGRE-14 WLS / Modal-WLS solver rows (unaffected by preprocessing)
- E2_corrected.csv            Oberrhein-70 comparison (100 rows)
- E3_corrected.csv            Bandwidth sensitivity (160 rows)
- E4_corrected.csv            Measurement-consistency weight sensitivity (100 rows)
- E5_corrected.csv            Robust-loss family comparison (120 rows)
- E6A_ieee30_corrected.csv    IEEE-30 comparison (60 rows)
- E6B_pantagruel_corrected.csv PanTaGruEl external benchmark (30 rows)
- e7_start_level.csv          Multi-start solver, per-start records (4000 rows)
- e7_selected_solution.csv    Multi-start solver, selected solutions (400 rows)

All neural results use preprocessing_version='train_only'. Raw schema:
experiment_id, dataset_id, data_source, external_benchmark, model, model_name,
architecture, loss, noise, bad_data_rate, bandwidth, bandwidth_schedule,
lambda_meas, eta, repeat_id, training_seed, data_seed, eval_noise_seed, epochs,
best_epoch, train_loss, val_loss, v_rmse, th_rmse, loading_rmse, modal_score,
hit_rate, runtime_seconds, converged, device, gpu_name, torch_version,
cuda_version, status, failure_epoch, failure_reason, v_mae, th_mae, state_loss,
meas_loss, notes.
