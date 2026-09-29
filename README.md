# Global Limits and Opportunities of Urban Trees for Reducing Cooling Energy Demand

## Tree states

The model is trained on four UT&C-BEM simulated tree-cover states for each of 100 cities:

* `notree`: no urban trees (`f_tree_mean = 0`, `LAI_TC_mean = 0`)
* `trees`: current tree cover (baseline)
* `plus10`: tree-cover fraction +10 percentage points
* `plus20`: tree-cover fraction +20 percentage points

## Contents

* `a1_merge_target_features_AllTreeLevels.py`: merges the ECAC targets and input features for all four tree states into one stacked training table
* `a2_XGB_model_training_AllTreeLevels.py`: XGBoost surrogate model tuning (day-CV and city-CV) and final model training
* `Extrapolation_validation_XBG_model.py`: validation of model extrapolation beyond the trained tree-cover range
* `UTC_BEM_Emulator_Input_DailySummer_AllTreeLevels_WEIGHTED.csv`: input features from UT&C-BEM for 100 cities and four tree states (CSV format)
* `SummerDaily_ECAC_100cities_AllTreeLevels_TEST_SEED.mat`: summer daily ECAC data from UT&C-BEM for 100 cities and four tree states
* `xgb_STACKED_AllTreeLevels_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700.joblib`: trained XGBoost model (final, all data, 700 estimators)
* `xgb_STACKED_AllTreeLevels_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700_TRAINFIT_METRICS.json`: training fit performance metrics
* `xgb_STACKED_AllTreeLevels_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700_feature_columns.json`: feature column names used in model training
* `selected_n_estimators_summary_STACKED.json`: n_estimators tuning summary and model configuration

## Dependencies

* Python 3.10.19
* numpy 2.2.6
* pandas 2.3.3
* h5py 3.13.0
* joblib 1.5.3
* xgboost 3.1.3
* scikit-learn 1.7.2
* matplotlib (for `Extrapolation_validation_XBG_model.py`)
