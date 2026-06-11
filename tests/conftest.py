import pandas as pd
import pytest

from wc26.data import ingest
from wc26.paths import processed_dir


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    if not (processed_dir() / "matches.parquet").exists():
        ingest.run()
    return ingest.load_matches(played_only=True)


@pytest.fixture(scope="session")
def shootouts() -> pd.DataFrame:
    if not (processed_dir() / "shootouts.parquet").exists():
        ingest.run()
    return ingest.load_shootouts()
