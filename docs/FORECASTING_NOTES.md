# Forecasting Methodology Notes

This project treats retail demand forecasting as a time-series problem rather than a standard randomly shuffled regression task.

## Time-aware validation

Random train-test splits can leak future information into model evaluation. `TimeSeriesSplit` preserves chronological ordering so validation data occurs after training data.

## Lag features

Historical sales provide important predictive signals. The project uses lagged sales features such as:

- Previous week
- Four weeks earlier
- Twelve weeks earlier

These features help the model capture short-term momentum and recurring demand patterns.

## Evaluation metrics

**MAE** provides an interpretable average absolute error.

**RMSE** penalizes larger forecast misses more strongly.

**MAPE** expresses error as a percentage, which is convenient for comparing stores but should be interpreted carefully when actual values approach zero.

## Store-level modeling

Separate models allow each store to have its own demand behavior. This also makes it easier to identify stores with unusually volatile sales or weaker forecast performance.

## Possible extensions

- Add richer holiday and promotion features
- Compare gradient boosting and statistical forecasting models
- Add prediction intervals
- Monitor forecast bias by store
- Add automated retraining as new weekly observations arrive
