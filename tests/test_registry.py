from wc26.data.registry import canonical, confed_map, is_registered, load_registry


def test_registry_loads_and_is_unique():
    df = load_registry()
    assert df.name.is_unique
    assert len(df) > 200


def test_aliases_canonicalize():
    assert canonical("USA") == "United States"
    assert canonical("Côte d'Ivoire") == "Ivory Coast"
    assert canonical("Türkiye") == "Turkey"
    assert is_registered("Korea Republic")


def test_wc2026_team_coverage(matches):
    # Every team in WC 2026 qualifying-era matches must be registered.
    cmap = confed_map()
    assert "Jordan" in cmap and cmap["Jordan"] == "AFC"
    assert cmap["Cape Verde"] == "CAF"
    assert cmap["Uzbekistan"] == "AFC"


def test_post2000_coverage_after_filter(matches):
    # The ingest filter drops unregistered teams; what remains must be large
    # and fully mapped (no silent confederation gaps).
    recent = matches[matches.date >= "2000-01-01"]
    assert len(recent) > 20000
    cmap = confed_map()
    teams = set(recent.home) | set(recent.away)
    unmapped = [t for t in teams if t not in cmap]
    assert not unmapped, f"unmapped teams slipped through ingest: {unmapped[:10]}"
