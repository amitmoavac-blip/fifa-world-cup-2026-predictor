import json

import pytest

from wc26.paths import processed_dir
from wc26.ui.build import ARTIFACT, build_artifact


@pytest.fixture(scope="module")
def client(matches):
    if not (processed_dir() / ARTIFACT).exists():
        build_artifact(write_snapshots=False)
    from wc26.ui.app import create_app

    return create_app().test_client()


@pytest.fixture(scope="module")
def artifact():
    return json.loads((processed_dir() / ARTIFACT).read_text())


def test_all_routes_200(client, artifact):
    for path in ["/", "/evaluation", "/health"]:
        assert client.get(path).status_code == 200
    mid = next(m["id"] for m in artifact["matches"] if m["status"] != "pending")
    assert client.get(f"/match/{mid}").status_code == 200
    pend = next(m["id"] for m in artifact["matches"] if m["status"] == "pending")
    assert client.get(f"/match/{pend}").status_code == 200


def test_unknown_match_404(client):
    assert client.get("/match/not-a-real-match").status_code == 404


def test_matches_page_shows_prediction(client):
    html = client.get("/").data.decode()
    assert "Predicted" in html
    assert "conf-" in html             # a confidence badge rendered
    assert "status-pending" in html    # knockout slots shown honestly


def test_filter_narrows_results(client):
    all_rows = client.get("/").data.decode().count('class="match-cell"')
    grp = client.get("/?stage=group").data.decode().count('class="match-cell"')
    one_team = client.get("/?team=Mexico").data.decode().count('class="match-cell"')
    assert grp < all_rows           # knockout slots filtered out
    assert 0 < one_team < grp       # Mexico appears in only a few fixtures


def test_match_detail_has_analyst_and_history(client, artifact):
    mid = next(m["id"] for m in artifact["matches"] if m["status"] != "pending")
    h = client.get(f"/match/{mid}").data.decode()
    assert "Predicted final score" in h
    assert "Top internal scorelines" in h
    assert "Prediction history" in h
    assert "not configured" in h      # honest about live/lineup epochs


def test_evaluation_shows_baselines_and_honest_framing(client):
    e = client.get("/evaluation").data.decode()
    assert "Elo-Poisson baseline" in e
    assert "Constant modal score" in e
    assert "not a measured ceiling" in e   # softened claim, no overclaim
    assert "status-disabled" in e          # experimental layers shown as off


def test_health_is_honest_about_live(client):
    hh = client.get("/health").data.decode()
    assert "not configured" in hh
    assert "historical / cached" in hh
