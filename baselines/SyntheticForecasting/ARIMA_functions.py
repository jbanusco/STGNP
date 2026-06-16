import pmdarima as pm
import numpy as np
from joblib import Parallel, delayed


def fit_predict_arima_per_series(ts, forecast_steps=1):
    try:
        # Use fixed order to avoid long auto_arima tuning
        # model = pm.ARIMA(order=(1, 0, 0), suppress_warnings=True)  # AR(1)
        model = pm.auto_arima(
            ts,
            start_p=1, max_p=3,
            start_q=0, max_q=2,
            d=None, max_d=1,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        model.fit(ts)
        forecast = model.predict(n_periods=forecast_steps)
        return forecast
    except Exception as e:
        # Fallback to naive prediction if ARIMA fails
        print('EH')
        return np.repeat(ts[-1], forecast_steps)


def arima_baseline(data, forecast_steps=1, n_jobs=-1):
    N, D, R, T = data.shape
    results = np.zeros((N, D, R, forecast_steps))

    def process_single(n, d, r):
        ts = data[n, d, r, :]  # Shape: [T]
        return fit_predict_arima_per_series(ts, forecast_steps)

    # Create list of jobs
    jobs = [(n, d, r) for n in range(N) for d in range(D) for r in range(R)]

    # Run in parallel
    forecasts = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single)(n, d, r) for (n, d, r) in jobs
    )

    # Refill results array
    for idx, (n, d, r) in enumerate(jobs):
        results[n, d, r, :] = forecasts[idx]

    return results  # Shape: [N, D, R, forecast_steps]