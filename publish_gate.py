"""
Whether a freshly generated plan should replace the published one.

`actions-gh-pages` replaces the entire publish directory, so publishing is
destructive: a fixture-blind plan does not sit alongside the good one, it
deletes it. But refusing to publish is not automatically safe either. The
published plan carries no expiry, so "keep the last good plan" and "silently
serve something from a previous gameweek" look identical to whoever reads it -
which is exactly how a plan for a gameweek that had already been played came to
be read as current.

So the rule is by gameweek rather than by quality alone. A sound plan always
publishes. A degraded one only holds back when what is already published is
sound *and for the same gameweek* - genuinely still usable. Once the published
plan belongs to a gameweek that has passed it is worse than an honest degraded
one, because a degraded plan announces itself and a stale plan does not.
"""
import json
import sys
from typing import Any, Dict, Optional, Tuple


def should_publish(new_plan: Dict[str, Any],
                   published: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """(publish?, reason). `published` is None when nothing is live yet."""
    if not new_plan.get("degraded"):
        return True, "forecast is sound"

    if not published:
        return True, ("nothing is published yet; a flagged degraded plan beats "
                      "no plan at all")

    if published.get("degraded"):
        return True, ("the published plan is also degraded, so there is no good "
                      "plan to protect")

    new_gw, old_gw = new_plan.get("gameweek"), published.get("gameweek")
    if new_gw is not None and old_gw == new_gw:
        return False, (f"keeping the sound plan already published for GW{old_gw}, "
                       f"which is still the gameweek in play")

    return True, (f"the published plan is for GW{old_gw} and we are now planning "
                  f"GW{new_gw}, so it is stale; publishing the degraded plan, "
                  f"which at least says so")


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        # No published plan, or an unreadable one: treat as nothing live.
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: publish_gate.py NEW_PLAN [PUBLISHED_PLAN]", file=sys.stderr)
        return 2
    new_plan = _load(sys.argv[1])
    if new_plan is None:
        # Nothing was generated, so there is nothing to publish - and crucially
        # nothing to overwrite the live plan with.
        print("publish=false")
        print("reason=no plan was generated")
        return 0
    ok, why = should_publish(new_plan, _load(sys.argv[2] if len(sys.argv) > 2 else None))
    print(f"publish={'true' if ok else 'false'}")
    print(f"reason={why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
