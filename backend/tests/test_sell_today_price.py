import datetime

import pytest

from backend.app.config import settings
from backend.app.services.compare import compare_mandis
from backend.app.services.forecast import generate_forecast, get_today_price
from backend.app.services.net_return import calculate_scenarios


@pytest.fixture()
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test_sell_today.duckdb")
    old_path = settings.DUCKDB_PATH
    settings.DUCKDB_PATH = db_path
    try:
        from backend.scripts.init_db import init_db

        init_db()
        yield db_path
    finally:
        settings.DUCKDB_PATH = old_path


def _insert_prices(db_path, crop_id, mandi_id, start, prices):
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        for i, price in enumerate(prices):
            day = start + datetime.timedelta(days=i)
            conn.execute(
                """INSERT INTO price_observations
                   (obs_id, crop_id, mandi_id, observation_date, min_price,
                    modal_price, max_price, arrivals_quintals, is_outlier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, False)""",
                (
                    f"{mandi_id}-{i}",
                    crop_id,
                    mandi_id,
                    day,
                    price * 0.9,
                    price,
                    price * 1.1,
                    100.0,
                ),
            )
    finally:
        conn.close()


def _horizon_zero_params():
    # vashi -> vashi has no distance row, so compare_mandis falls back to
    # 50 km at 40 km/h; tomato cost params come from init_db seed data.
    return {
        "s_transit": (50.0 / 40.0) * 0.005,
        "s_hold": 0,
        "commission_rate": 0.06,
        "transport_cost_per_quintal": 50.0 * 2.0,
        "surcharge_per_quintal": 0,
        "rain_mult": 1.0,
    }


def test_sell_today_uses_last_observed_price(seeded_db):
    # Rising series with noise: 20 days, last observed modal price is 1195.
    prices = [1000 + 10 * i + (5 if i % 2 else -5) for i in range(20)]
    start = datetime.date(2023, 1, 1)
    _insert_prices(seeded_db, "tomato", "vashi", start, prices)
    origin = "2023-01-20"
    last_observed = prices[-1]
    assert last_observed == 1195

    # The forecast itself must stay 1..N: forecasts[0] is tomorrow.
    forecasts = generate_forecast("tomato", "vashi", origin, horizon_days=2)
    assert forecasts[0]["horizon"] == 1
    # EWMA lags a rising series, so tomorrow's forecast differs from today.
    assert forecasts[0]["p50"] != last_observed

    # Today row must use today's actual price, with a real band.
    today = get_today_price("tomato", "vashi", origin)
    assert today["horizon"] == 0
    assert today["p50"] == last_observed
    assert today["p10"] < today["p50"] < today["p90"]


@pytest.mark.asyncio
async def test_compare_sell_today_row_priced_at_today(seeded_db):
    prices = [1000 + 10 * i + (5 if i % 2 else -5) for i in range(20)]
    start = datetime.date(2023, 1, 1)
    _insert_prices(seeded_db, "tomato", "vashi", start, prices)
    origin = "2023-01-20"
    last_observed = prices[-1]

    res = await compare_mandis("tomato", "vashi", 50, origin, 2)
    row0 = next(
        r for r in res["results"] if r["mandi_id"] == "vashi" and r["horizon"] == 0
    )

    # Real uncertainty band on the sell-today row, not a single point.
    assert row0["nr_p10"] < row0["nr_p50"] < row0["nr_p90"]

    # Sell-today net return must come from today's price, not tomorrow's.
    expected = calculate_scenarios(
        last_observed, last_observed, last_observed, 50, _horizon_zero_params()
    )["p50"]["net_return_per_quintal"]
    assert row0["nr_p50"] == expected


@pytest.mark.asyncio
async def test_compare_fallback_path_still_works(seeded_db):
    # Fewer than 7 days of history: fallback path.
    prices = [1000, 1010, 1020]
    start = datetime.date(2023, 1, 1)
    _insert_prices(seeded_db, "tomato", "vashi", start, prices)
    origin = "2023-01-03"
    last_observed = prices[-1]

    today = get_today_price("tomato", "vashi", origin)
    assert today["p50"] == last_observed
    assert today["p10"] < today["p50"] < today["p90"]

    res = await compare_mandis("tomato", "vashi", 50, origin, 2)
    row0 = next(
        r for r in res["results"] if r["mandi_id"] == "vashi" and r["horizon"] == 0
    )
    assert row0["nr_p10"] < row0["nr_p50"] < row0["nr_p90"]
    expected = calculate_scenarios(
        last_observed, last_observed, last_observed, 50, _horizon_zero_params()
    )["p50"]["net_return_per_quintal"]
    assert row0["nr_p50"] == expected
