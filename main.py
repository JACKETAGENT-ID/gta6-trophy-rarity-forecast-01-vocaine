"""GTA VI trophy rarity forecast (vocaine entry, round 2) -- earners/players
decomposition with a hazard + marginal-earner model.

A trophy's rate is a ratio of two counts: E (players tracked who earned it)
over P (players tracked).  Both counts only grow.  The forecast projects
both to the target and divides:

    rate_target = 100 * E_target / P_target

Two forces move them:

  * Progression: players already tracked keep earning, at a per-trophy
    hourly hazard lam_i on the not-yet-earners (P - E).
  * Dilution: new players arrive (P grows) having earned almost nothing
    yet; each newcomer has earned trophy i with marginal rate m_i.

Stepping hourly from the checkpoint to the target:

    E_i += lam_i * (P - E_i) * 1h + m_i * dP
    P   <- projected players-tracked (log-time fit, see below)

The case's own history decides which regime runs:

  * FLAT history (every read identical, as on launch night): no refresh
    has been observed, so the next refresh is the whole story.  The kit
    documents the first refresh exactly: 638 players tracked and 943
    trophies earned became 3,868 players and 4,276 trophies earned, every
    rate but the one at 100% lower.  So P jumps ~6.06x, and the newcomers
    earned ~103 trophies between them -- almost all on trophies a new
    player can earn in their first hours (the story path and the quickest
    one-off activities), almost none on grinds, late-story strands or the
    platinum.  The median models that one jump explicitly; the spread
    carries the risk that the refresh lands differently.
  * TRENDING history (the final case): lam_i is measured from stretches
    where P is stable, m_i from the jumps where P refreshes, both shrunk
    toward class priors; P is projected by a log-time fit over the last
    24h of reads.  The median is the logit-mean of this ODE forecast and
    a plain logit-trend forecast (least squares of logit(rate) on
    ln(1 + hours since release) over the last 24h), and the spread
    carries their disagreement -- the two disagree exactly where the
    data is thinnest.

A trophy at 100% in every read stays ~100%: the first refresh left it
there, and dilution cannot pull a saturated trophy down.

Python 3.13, stdlib only, deterministic: no network, no clock, no
randomness.  Reads every *.in in MAKEFUN_INPUT_DIR (default /input),
writes <id>.out to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- First refresh, as the kit documents it (RULES / competition.yaml) ---
# Launch night: 638 players tracked, 943 trophies earned; refreshed to
# 3,868 players, 4,276 trophies earned.
FIRST_P = 638.0
REFRESH_P = 3868.0
REFRESH_GROWTH = REFRESH_P / FIRST_P  # ~6.06x players at a refresh
# Newcomers' earn-share: a new player earns an early trophy at about this
# fraction of the old cohort's rate (calibrated on the practice cases:
# the refresh's 103 extra earners sit on the most-earned trophies, and
# the fitted answers put the newcomer share near 0.14-0.16).
NEW_SHARE = 0.15

# --- Spread ---
SIGMA_RW = 0.35        # random-walk logit sigma per sqrt(day)
SIGMA_REFRESH = 0.10   # extra logit sigma while the next refresh is unseen
SIGMA_BASE = 0.02      # floor variance term

# --- Trend regime ---
TREND_WINDOW_HOURS = 24.0
MAX_PLAYER_MULT = 8.0
ODE_TREND_MIX = 0.5    # weight of the ODE forecast in the logit-mean

# Trophy classes: (hourly hazard prior, newcomer-share weight w).
# w is the share of the newcomer earn-share a trophy gets at a refresh:
# 1.0 the story path / quick one-offs, 0.30 the final mission,
# 0.15 grinds, side strands and ordinary activities, 0 the platinum.
# (Calibrated on the practice cases: the refresh's extra earners sit
#  almost entirely on the story path; one-off activities and side
#  content gain ~nothing in a newcomer's first hours.)
_GRIND_WORDS = (
    "100% completion", "250 stunts", "gold medal in all",
    "every location", "every spot", "100 locations",
    "collectible card sets", "50 challenges", "5 bank vaults",
    "8 mission strands", "20 different races", "each gang",
    "all garage properties", "$250,000",
)
_QUICK_WORDS = (
    "fishing challenge", "arcade game",
)
# class: (hazard prior per hour, w)
_CLASS = {
    "plat": (0.00002, 0.0),
    "grind": (0.0003, 0.10),
    "late": (0.001, 0.30),
    "mid": (0.003, 0.15),
    "early": (0.008, 1.0),
    "misc": (0.005, 0.15),
}


def _parse_time(text):
    if text is None:
        return None
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct)
    return math.log(p / (100.0 - p))


def _inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _classify(trophy):
    """Trophy class from tier + description keywords."""
    if trophy.get("tier") == "platinum":
        return "plat"
    hay = ((trophy.get("name") or "") + " " +
           (trophy.get("description") or "")).lower()
    desc = (trophy.get("description") or "").lower()
    if any(w in hay for w in _GRIND_WORDS):
        return "grind"
    if "final mission" in desc and " for " not in desc:
        return "late"
    if any(w in hay for w in _QUICK_WORDS):
        return "early"
    if desc.startswith("complete ") and " for " not in desc \
            and "strands" not in desc:
        return "early"
    if "chapter" in desc:
        return "early"
    if desc.startswith("complete ") or "mission" in desc:
        return "mid"
    return "misc"


def _fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_var, sxx, n, xbar)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0, n, mx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid, sxx, n, mx


def _history_is_flat(history):
    last_rates = history[-1]["rates"]
    return all(read["rates"] == last_rates for read in history)


def _last_known_rate(history, i):
    for read in reversed(history):
        rate = read["rates"][i]
        if rate is not None:
            return rate
    return None


def _earners(history, i):
    """Earned counts E_i(t) per read (None where the read lacked the rate)."""
    out = []
    for read in history:
        rate = read["rates"][i]
        if rate is None:
            out.append(None)
        else:
            out.append(rate / 100.0 * read["players_tracked"])
    return out


def _measure_hazard(history, i, cls):
    """Hourly hazard lam_i from stretches where P is stable.

    lam = dE / ((P - E) * dt), averaged over stable intervals, shrunk
    toward the class prior when the history shows little.
    """
    prior, _w = _CLASS[cls]
    earners = _earners(history, i)
    vals = []
    for prev, cur, e0, e1 in zip(history, history[1:], earners, earners[1:]):
        if e0 is None or e1 is None:
            continue
        p0, p1 = prev["players_tracked"], cur["players_tracked"]
        if p0 != p1 or p0 <= 0:
            continue
        t0 = _parse_time(prev["at"])
        t1 = _parse_time(cur["at"])
        dt = (t1 - t0).total_seconds() / 3600.0
        if dt < 0.5:
            continue
        pool = p0 - e0
        if pool <= 0:
            continue
        lam = (e1 - e0) / (pool * dt)
        if 0.0 <= lam <= 0.05:
            vals.append((lam, dt))
    if not vals:
        return prior
    num = sum(lam * dt for lam, dt in vals)
    den = sum(dt for _lam, dt in vals)
    measured = num / den if den > 0 else prior
    # Shrink: a couple of short intervals do not outweigh the prior.
    weight = min(1.0, den / 12.0)
    return weight * measured + (1.0 - weight) * prior


def _measure_marginal(history, i, cls, prior_m):
    """Newcomer earn rate m_i from refresh jumps: dE/dP, jump-weighted."""
    earners = _earners(history, i)
    num = 0.0
    den = 0.0
    for prev, cur, e0, e1 in zip(history, history[1:], earners, earners[1:]):
        if e0 is None or e1 is None:
            continue
        p0, p1 = prev["players_tracked"], cur["players_tracked"]
        if p1 <= p0:
            continue
        m = (e1 - e0) / (p1 - p0)
        if 0.0 <= m <= 1.0:
            num += m * (p1 - p0)
            den += (p1 - p0)
    if den <= 0:
        return prior_m
    measured = num / den
    weight = min(1.0, den / 5000.0)
    return weight * measured + (1.0 - weight) * prior_m


def _project_players(history, release, checkpoint, target):
    """Players-tracked at the target: log P linear in ln(1+hours)."""
    pts = []
    for read in history:
        t = _parse_time(read["at"])
        h = (t - release).total_seconds() / 3600.0
        p = read["players_tracked"]
        if p > 0 and h >= 0:
            pts.append((math.log1p(h), math.log(p)))
    last_p = history[-1]["players_tracked"]
    if len(pts) < 3 or len(set(x for x, _y in pts)) < 2:
        return float(last_p)
    # Last 24h of reads only (the log-time fit already damps the launch
    # spike, and the window keeps the projection on recent growth).
    cutoff_h = (checkpoint - release).total_seconds() / 3600.0 - TREND_WINDOW_HOURS
    window = []
    for read in history:
        t = _parse_time(read["at"])
        h = (t - release).total_seconds() / 3600.0
        p = read["players_tracked"]
        if h >= cutoff_h and p > 0 and h >= 0:
            window.append((math.log1p(h), math.log(p)))
    if len(window) >= 3 and len(set(x for x, _y in window)) >= 2:
        pts = window
    a, b, _r, _s, _n, _m = _fit_line([x for x, _y in pts], [y for _x, y in pts])
    ht = (target - release).total_seconds() / 3600.0
    proj = math.exp(a + b * math.log1p(max(0.0, ht)))
    return min(max(float(last_p), proj), float(last_p) * MAX_PLAYER_MULT)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    history = case["history"]
    last = history[-1]
    last_rates = last["rates"]
    p_last = float(last["players_tracked"])

    release = _parse_time((case.get("game") or {}).get("release_date"))
    checkpoint = _parse_time(case["at"])
    target = _parse_time(case["target"])
    if release is None:
        release = checkpoint
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0

    flat = _history_is_flat(history)
    classes = [_classify(t) for t in trophies]
    rows = []

    if flat:
        # --- Regime A: one first refresh before the target. ---
        p_target = p_last * REFRESH_GROWTH
        p_new = p_target - p_last
        sigma = math.sqrt(SIGMA_BASE + SIGMA_REFRESH ** 2
                          + (SIGMA_RW ** 2) * horizon_days)
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate is None:
                rate = _last_known_rate(history, i)
            if rate is None:
                rate = 0.5
            if rate >= 99.99 and all(
                    (r["rates"][i] or 0) >= 99.0 for r in history):
                rows.append({"id": trophy["id"],
                             "q": [99.99, 99.99, 99.99, 99.99, 100.0]})
                continue
            cls = classes[i]
            _lam, w = _CLASS[cls]
            count = rate / 100.0 * p_last
            share = (rate / 100.0) * NEW_SHARE * w
            e_target = count + p_new * share
            median = _clamp_pct(100.0 * e_target / p_target)
            centre = _logit(median)
            quantiles = []
            prev_q = 0.0
            for z in Z:
                q = _round4(_clamp_pct(_inv_logit(centre + z * sigma)))
                q = max(q, prev_q)
                quantiles.append(q)
                prev_q = q
            rows.append({"id": trophy["id"], "q": quantiles})
        return {"trophies": rows}

    # --- Regime B: trending history (the final case). ---
    p_target = _project_players(history, release, checkpoint, target)
    # P(t) path: log-linear in log1p(hours) from checkpoint to target.
    h0 = max(0.0, (checkpoint - release).total_seconds() / 3600.0)
    h1 = max(h0, (target - release).total_seconds() / 3600.0)
    steps = max(1, int(math.ceil(horizon_h)))
    log_p0 = math.log(p_last)
    log_p1 = math.log(max(p_last, p_target))
    # Precompute the P path (hourly).
    p_path = []
    for s in range(steps + 1):
        f = s / steps
        lh = math.log1p(h0 + f * (h1 - h0))
        lh0 = math.log1p(h0)
        lh1 = math.log1p(h1)
        if lh1 > lh0:
            g = (lh - lh0) / (lh1 - lh0)
        else:
            g = f
        p_path.append(math.exp(log_p0 + g * (log_p1 - log_p0)))

    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = _last_known_rate(history, i)
        if rate is None:
            rate = 0.5
        if rate >= 99.99 and all(
                (r["rates"][i] or 0) >= 99.0 for r in history):
            rows.append({"id": trophy["id"],
                         "q": [99.99, 99.99, 99.99, 99.99, 100.0]})
            continue
        cls = classes[i]
        lam_prior, w = _CLASS[cls]
        lam = _measure_hazard(history, i, cls)
        prior_m = (rate / 100.0) * NEW_SHARE * w
        marg = _measure_marginal(history, i, cls, prior_m)

        # ODE forecast: step E hourly along the P path.
        e_ode = rate / 100.0 * p_last
        for s in range(steps):
            p_now = p_path[s]
            p_nxt = p_path[s + 1]
            e_ode += lam * max(0.0, p_now - e_ode) * 1.0
            e_ode += marg * max(0.0, p_nxt - p_now)
            if e_ode > p_nxt:
                e_ode = p_nxt
        ode_rate = _clamp_pct(100.0 * e_ode / p_target)

        # Trend forecast: logit(rate) linear in ln(1+hours), last 24h.
        cutoff_h = (checkpoint - release).total_seconds() / 3600.0 \
            - TREND_WINDOW_HOURS
        xs, ys = [], []
        for read in history:
            t = _parse_time(read["at"])
            h = (t - release).total_seconds() / 3600.0
            r = read["rates"][i]
            if r is None or h < cutoff_h:
                continue
            xs.append(math.log1p(max(0.0, h)))
            ys.append(_logit(r))
        if len(xs) >= 3 and len(set(xs)) >= 2 and len(set(ys)) >= 2:
            a, b, resid, sxx, cnt, xbar = _fit_line(xs, ys)
            x_t = math.log1p(max(0.0, h1))
            centre_tr = a + b * x_t
            slope_var = 0.0
            if sxx > 1e-12 and cnt > 2 and resid > 0.0:
                slope_var = (resid / sxx) * (x_t - xbar) ** 2
            trend_rate = _clamp_pct(_inv_logit(centre_tr))
            trend_spread = math.sqrt(resid + slope_var)
        else:
            trend_rate = ode_rate
            trend_spread = 0.0

        # Logit-mean hedge; spread carries the disagreement.
        lo = _logit(ode_rate)
        lt = _logit(trend_rate)
        centre = ODE_TREND_MIX * lo + (1.0 - ODE_TREND_MIX) * lt
        move = abs(centre - _logit(rate))
        sigma = math.sqrt(
            SIGMA_BASE + (SIGMA_RW ** 2) * horizon_days
            + 0.01 + (trend_spread ** 2) * 0.25
            + (0.05 * move) ** 2 + (abs(lo - lt) / 4.0) ** 2)
        sigma = min(1.5, max(0.15, sigma))
        quantiles = []
        prev_q = 0.0
        for z in Z:
            q = _round4(_clamp_pct(_inv_logit(centre + z * sigma)))
            q = max(q, prev_q)
            quantiles.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
