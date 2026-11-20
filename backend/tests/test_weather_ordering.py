import os

import duckdb
import pytest

from backend.app.config import settings
from backend.app.ingestion.weather import get_weather
from backend.app.services.compare import compare_mandis


@pytest.fixture()
def seeded_db(tmp_path):
    """Point the app at a fresh DuckDB file with the full schema and seeds."""
    test_db = str(tmp_path / "weather_ordering.duckdb")
    old_path = settings.DUCKDB_PATH
    settings.DUCKDB_PATH = test_db
    try:
        from backend.scripts.init_db import init_db

        init_db()
        yield test_db
    finally:
        settings.DUCKDB_PATH = old_path


def _seed_weather(conn, location_id, rows):
    # rows: list of (date, precip_mm); inserted in the given order on purpose
    for date, precip in rows:
        conn.execute(
            "INSERT INTO weather_daily (location_id, date, precip_mm, precip_prob, temp_max, temp_min, is_forecast, source, fetched_at, data_origin)"
            " VALUES (?, ?, ?, 0, 30.0, 20.0, true, 'test', CURRENT_TIMESTAMP, 'synthetic')",
            (location_id, date, precip),
        )


def test_get_weather_returns_oldest_first(seeded_db):
    conn = duckdb.connect(seeded_db)
    try:
        # Inserted out of order, plus one stale row older than the window.
        _seed_weather(conn, "vashi", [
            ("2023-01-03", 25.0),
            ("2022-12-31", 99.0),
            ("2023-01-01", 0.0),
            ("2023-01-02", 0.0),
        ])
    finally:
        conn.close()

    res = get_weather("vashi", 3)

    assert res["status"] == "ok"
    dates = [str(row[0]) for row in res["weather"]]
    # Most recent 3-day window, oldest first: index h lines up with horizon h.
    assert dates == ["2023-01-01", "2023-01-02", "2023-01-03"]
    assert [row[1] for row in res["weather"]] == [0.0, 0.0, 25.0]


@pytest.mark.asyncio
async def test_rain_multiplier_lands_on_correct_horizon(seeded_db):
    conn = duckdb.connect(seeded_db)
    try:
        # Rain only on 2023-01-03 (horizon 2). The stale 2022-12-31 row
        # must not leak into the 3-day window.
        _seed_weather(conn, "vashi", [
            ("2023-01-03", 25.0),
            ("2022-12-31", 99.0),
            ("2023-01-01", 0.0),
            ("2023-01-02", 0.0),
        ])
    finally:
        conn.close()

    rainy = await compare_mandis("tomato", "vashi", 50, "2023-01-01", 3)

    conn = duckdb.connect(seeded_db)
    try:
        conn.execute("DELETE FROM weather_daily WHERE location_id = 'vashi'")
        _seed_weather(conn, "vashi", [
            ("2023-01-01", 0.0),
            ("2023-01-02", 0.0),
            ("2023-01-03", 0.0),
        ])
    finally:
        conn.close()

    dry = await compare_mandis("tomato", "vashi", 50, "2023-01-01", 3)

    def by_horizon(res, mandi_id):
        return {r["horizon"]: r["nr_p50"] for r in res["results"] if r["mandi_id"] == mandi_id}

    rainy_vashi = by_horizon(rainy, "vashi")
    dry_vashi = by_horizon(dry, "vashi")

    # Rain on horizon 2 only: horizons 0 and 1 must be untouched...
    assert rainy_vashi[0] == dry_vashi[0]
    assert rainy_vashi[1] == dry_vashi[1]
    # ...and horizon 2 must carry the rain cost.
    assert rainy_vashi[2] < dry_vashi[2]
