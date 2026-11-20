import numpy as np

from backend.app.config import settings
from backend.app.services.series import get_series


def _residual_percentiles(df) -> tuple:
    """10th/90th percentiles of relative one-step EWMA forecast errors."""
    df['ewma_pred'] = df['modal_price'].ewm(alpha=settings.EWMA_ALPHA, adjust=False).mean().shift(1)
    res_rel = ((df['modal_price'] - df['ewma_pred']) / df['ewma_pred']).dropna()

    if len(res_rel) >= settings.MIN_RESIDUALS_FOR_EMPIRICAL_BAND:
        return np.percentile(res_rel, 10), np.percentile(res_rel, 90)
    return -0.20, 0.20


def get_today_price(crop_id: str, mandi_id: str, origin_date: str) -> dict:
    """Today's actual price: the last observed modal price.

    Returns a horizon 0 row. Unlike the forecast rows (horizons 1..N),
    this is not a prediction: p50 is the last observed modal price, and
    the p10/p90 band comes from the spread of past one-step forecast
    errors, so selling today still shows real uncertainty.
    """
    df = get_series(crop_id, mandi_id, origin_date)

    if df.empty:
        last_val = 1000.0
        return {"horizon": 0, "p10": last_val * 0.8, "p50": last_val, "p90": last_val * 1.2}

    last_val = float(df['modal_price'].iloc[-1])

    if len(df) < 7:
        # Same +/-20% band as the fallback forecast path.
        return {"horizon": 0, "p10": last_val * 0.8, "p50": last_val, "p90": last_val * 1.2}

    p10_pct, p90_pct = _residual_percentiles(df)
    # Symmetric band around the observed price, sized by the spread of
    # past forecast errors; p50 stays exactly the last observed price.
    half_width = (p90_pct - p10_pct) / 2
    p50 = max(1.0, last_val)
    p10 = max(1.0, p50 * (1 - half_width))
    p90 = max(1.0, p50 * (1 + half_width))

    return {"horizon": 0, "p10": p10, "p50": p50, "p90": p90}


def generate_forecast(crop_id: str, mandi_id: str, origin_date: str, horizon_days: int = 7) -> list:
    df = get_series(crop_id, mandi_id, origin_date)

    if len(df) < 7:
        last_val = df['modal_price'].iloc[-1] if not df.empty else 1000.0
        return [{"horizon": h, "p10": last_val*0.8, "p50": last_val, "p90": last_val*1.2, "method": "fallback", "calibrated": False} for h in range(1, horizon_days+1)]

    alpha = settings.EWMA_ALPHA
    ewma = df['modal_price'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    recent = df['modal_price'].tail(14).values
    trend = np.mean(np.diff(recent)) if len(recent) > 1 else 0
    damping = settings.TREND_DAMPING

    p10_pct, p90_pct = _residual_percentiles(df)

    forecasts = []
    current_level = ewma
    current_trend = trend

    for h in range(1, horizon_days + 1):
        current_trend *= damping
        current_level += current_trend

        p50 = max(1.0, current_level)
        p10 = max(1.0, p50 * (1 + p10_pct))
        p90 = max(1.0, p50 * (1 + p90_pct))

        p10, p50, p90 = sorted([p10, p50, p90])

        forecasts.append({
            "horizon": h,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "method": "baseline_ewma_damped_trend",
            "calibrated": False
        })

    return forecasts
