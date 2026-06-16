# Global Limits and Opportunities of Urban Trees for Reducing Cooling Energy Demand


## Contents

- **XGB_model_training.py** — XGBoost surrogate model training code
- **UTC_BEM_Emulator_Input_DailySummer_NEW_WEIGHTED.csv** — Input training data (CSV format)
- **UTC_BEM_Emulator_Input_DailySummer_NEW_WEIGHTED.mat** — Input training data (MATLAB format)
- **SummerDaily_ECAC_PYFRIENDLY_Tropical_fixed_WEIGHTED.mat** — Summer daily cooling energy data for tropical cities
- **xgb_STACKED_TreesPlusNoTree_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700.joblib** — Trained XGBoost model (final, all data, 700 estimators)
- **xgb_STACKED_TreesPlusNoTree_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700_TRAINFIT_METRICS.json** — Training fit performance metrics
- **xgb_STACKED_TreesPlusNoTree_CITYFE_dualmonotone_noHemisphere_FINAL_ALLDATA_n700_feature_columns.json** — Feature column names used in model training

## Dependencies

- Python 3.10.19
- numpy 2.2.6
- pandas 2.3.3
- h5py 3.13.0
- joblib 1.5.3
- xgboost 3.1.3
- scikit-learn 1.7.2

