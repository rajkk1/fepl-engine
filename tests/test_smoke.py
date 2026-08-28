import pytest

def test_smoke_engine():
    import weekly_manager
    from xp_model import generate_merv_matrix
    from fpl_api import get_bootstrap_static, get_fixtures
    
    # 1. Test weekly_manager arity
    res = weekly_manager.get_manager_team_state(12345, 1)
    assert len(res) == 5, f"Arity crash: returned {len(res)} values"
    
    # 2. Test generate_xp_matrix end-to-end (mock logic)
    bootstrap = get_bootstrap_static(use_cache=True)
    if bootstrap:
        # Just grab two players to run it
        bootstrap["elements"] = bootstrap["elements"][:50]
        fixtures = get_fixtures(use_cache=True)
        xp_matrix = generate_merv_matrix([1, 2], bootstrap=bootstrap, fixtures=fixtures)
        
        for pid in xp_matrix:
            assert xp_matrix[pid].get(1, -1) >= 0.0, "xP should be finite and positive"
            assert f"1_p_play" in xp_matrix[pid], "p_play should be emitted in the matrix"
