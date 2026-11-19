"""Prakiraan rarity trophy GTA VI (Exophase) -- model kurva tier empiris.

Ide: rate trophy pada target adalah campuran dua kohort pemain.
Kohort lama (yang sudah terlacak di checkpoint) hanya bisa menambah
peraih, jadi ratenya di dalam kohort itu tidak pernah turun; ia merangkak
menuju "prior minggu-1" trophy itu dengan kecepatan yang bergantung pada
kategori trophy (story awal cepat, grind/koleksi lambat). Kohort baru
(pemain yang bergabung antara checkpoint dan target) mulai dari nol dan
mengejar prior yang sama, tapi lebih lambat karena waktu bermain mereka
lebih singkat. Rate gabungan = rata-rata tertimbang kedua kohort menurut
jumlah pemain masing-masing.

Prior minggu-1 ditetapkan dari tier + isi deskripsi trophy (posisi chapter
untuk story, kata kunci grind/koleksi untuk trophy lambat, platinum sangat
rendah), lalu tiap trophy dikalibrasi terhadap rate terakhir yang terbaca:
faktor kalibrasi = rate terakhir / rate yang diprediksi model pada
checkpoint, dijepit agar satu trophy aneh tidak meledakkan prakiraan.

Jumlah pemain di target: bila histori sudah memuat refresh (jumlah pemain
melonjak), laju pertumbuhan jam-jaman yang teramati diekstrapolasi; bila
belum ada refresh sama sekali dan checkpoint masih dalam 24 jam pertama
sejak rilis, diasumsikan satu refresh malam-peluncuran terjadi sebelum
target (pemain x6,06 -- 638 -> 3.868 seperti yang didokumentasikan RULES:
semua rate kecuali yang 100% turun saat refresh itu). Sensitivitas tiap
trophy terhadap gelombang pemain baru juga dibaca dari refresh yang
teramati di histori bila ada: trophy yang ratenya jatuh paling dalam saat
refresh dianggap paling lambat dikejar kohort baru.

Spread: kuantil dihitung pada skala logit di sekitar median, dengan sigma
yang membesar bersama akar horizon dan lebih lebar untuk trophy langka /
grind; bila belum ada refresh teramati, ditambah ketidakpastian refresh.

Python 3.13, stdlib saja, deterministik, tanpa network / random / waktu.
Membaca setiap *.in di MAKEFUN_INPUT_DIR (default /input), menulis
<id>.out ke MAKEFUN_OUTPUT_DIR (default /output).
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

# Satu refresh malam peluncuran: 638 -> 3.868 pemain (RULES.txt).
FIRST_REFRESH_MULT = 3868.0 / 638.0
# Batas faktor kalibrasi per trophy.
CAL_MIN, CAL_MAX = 0.15, 5.0
# Ketidakpastian logit tambahan bila histori belum memuat satu refresh pun.
UNSEEN_REFRESH_SIGMA = 0.45


def _parse_time(text):
    """ISO 8601 (dengan 'Z' atau tanggal saja) -> datetime UTC."""
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
    """Bulatkan ke 4 desimal, half-up; bilangan bulat ditulis tanpa .0."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


# (kata kunci deskripsi -> (prior minggu-1 %, tau hari, sigma dasar, sens default))
# Dicek berurutan; yang pertama cocok menang.
_KEYWORD_RULES = [
    ("100% completion", (0.35, 10.0, 0.85, 1.00)),
    ("chapter 1", (38.0, 1.0, 0.45, 0.85)),
    ("chapter 2", (30.0, 1.5, 0.45, 0.90)),
    ("chapter 3", (24.0, 2.0, 0.50, 0.90)),
    ("chapter 4", (19.0, 2.5, 0.50, 0.95)),
    ("final mission for", (7.0, 2.5, 0.55, 0.90)),
    ("final mission", (3.0, 3.0, 0.60, 0.95)),
    ("250 stunts", (0.8, 6.0, 0.80, 1.00)),
    ("gold medal in all", (0.7, 6.0, 0.80, 1.00)),
    ("every location", (1.0, 5.0, 0.80, 1.00)),
    ("every spot", (1.2, 5.0, 0.80, 1.00)),
    ("50 challenges", (1.5, 5.0, 0.80, 1.00)),
    ("collectible card sets", (0.9, 6.0, 0.80, 1.00)),
    ("100 locations", (0.6, 6.0, 0.80, 1.00)),
    ("5 bank vaults", (1.5, 4.0, 0.75, 1.00)),
    ("8 mission strands", (2.0, 4.0, 0.75, 0.95)),
    ("100 vehicles", (2.5, 4.0, 0.75, 1.00)),
    ("20 different races", (2.0, 4.0, 0.75, 0.95)),
    ("each type of activity", (8.0, 2.0, 0.60, 0.85)),
    ("each gang", (2.0, 4.0, 0.75, 1.00)),
    ("all of leonida", (3.0, 4.0, 0.70, 0.95)),
    ("all garage properties", (1.5, 5.0, 0.80, 1.00)),
    ("$250,000", (2.0, 4.0, 0.75, 1.00)),
    ("$100,000", (6.0, 3.0, 0.65, 0.90)),
    ("10 vehicles", (3.5, 3.0, 0.65, 0.90)),
]

# Default per tier bila tak ada kata kunci cocok: (prior %, tau hari, sigma, sens).
_TIER_DEFAULT = {
    "platinum": (0.30, 7.0, 1.00, 1.00),
    "gold": (2.00, 3.0, 0.85, 0.95),
    "silver": (1.20, 5.0, 0.80, 1.00),
    "bronze": (4.00, 2.5, 0.60, 0.90),
}


def _classify(trophy):
    """Kembalikan (prior, tau, sigma_dasar, sens_default) untuk satu trophy."""
    desc = (trophy.get("description") or "").lower()
    for key, rule in _KEYWORD_RULES:
        if key in desc:
            prior, tau, sigma, sens = rule
            break
    else:
        # Trophy story generik ("Complete <misi>...") tanpa chapter.
        if desc.startswith("complete "):
            prior, tau, sigma, sens = 12.0, 2.0, 0.55, 0.90
        else:
            prior, tau, sigma, sens = _TIER_DEFAULT.get(
                trophy.get("tier"), _TIER_DEFAULT["bronze"]
            )
    if trophy.get("secret") and "chapter" not in desc:
        prior *= 0.85
    return prior, tau, sigma, sens


def _refresh_events(history, n_trophies):
    """Pasangan read berurutan di mana jumlah pemain berubah.

    Mengembalikan list (pemain_sebelum, pemain_sesudah, rates_sebelum,
    rates_sesudah). Trophy yang ratenya jatuh paling dalam di event ini
    adalah yang paling sensitif terhadap gelombang pemain baru.
    """
    events = []
    for prev, cur in zip(history, history[1:]):
        p0, p1 = prev["players_tracked"], cur["players_tracked"]
        if p1 != p0 and prev["rates"] is not None and cur["rates"] is not None:
            events.append((p0, p1, prev["rates"], cur["rates"]))
    return events


def _sensitivities(events, n_trophies, defaults):
    """Sensitivitas per trophy 0..1 dari refresh teramati, atau default."""
    if not events:
        return list(defaults)
    sens = []
    for i in range(n_trophies):
        vals = []
        for p0, p1, r0, r1 in events:
            a, b = r0[i], r1[i]
            if a is None or b is None or p1 <= p0 or a <= 0:
                continue
            # Rate kohort baru yang tersirat oleh refresh ini.
            new_rate = (b * p1 - a * p0) / (p1 - p0)
            vals.append(min(1.0, max(0.0, 1.0 - new_rate / a)))
        sens.append(sum(vals) / len(vals) if vals else defaults[i])
    return sens


def _expected_players(case, history, events):
    """Perkiraan jumlah pemain terlacak pada target."""
    last_players = history[-1]["players_tracked"]
    horizon_h = case["horizon_minutes"] / 60.0
    if events:
        first, last = history[0], history[-1]
        t0 = _parse_time(first["at"])
        t1 = _parse_time(last["at"])
        span_h = max(0.5, (t1 - t0).total_seconds() / 3600.0)
        growth = math.log(max(1.0, last_players) / max(1.0, first["players_tracked"]))
        rate_h = growth / span_h
        mult = math.exp(rate_h * horizon_h)
        return last_players * min(40.0, max(1.0, mult))
    release = _parse_time((case.get("game") or {}).get("release_date"))
    checkpoint = _parse_time(case["at"])
    if release is not None and checkpoint is not None:
        age_h = (checkpoint - release).total_seconds() / 3600.0
        if 0 <= age_h <= 24.0:
            # Belum ada refresh teramati: satu refresh malam peluncuran
            # diasumsikan mendarat sebelum target.
            return last_players * FIRST_REFRESH_MULT
    return last_players * (1.0 + 0.35 * (horizon_h / 24.0))


def _matured(prior, tau_days, days):
    """Fraksi prior yang dicapai kohort berumur `days` hari."""
    return prior * (1.0 - math.exp(-max(0.0, days) / tau_days))


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    last = case["history"][-1]
    last_rates = last["rates"]
    last_players = last["players_tracked"]

    release = _parse_time((case.get("game") or {}).get("release_date"))
    checkpoint = _parse_time(case["at"])
    target = _parse_time(case["target"])
    if release is None:
        release = checkpoint
    days_check = max(0.0, (checkpoint - release).total_seconds() / 86400.0)
    days_target = max(0.0, (target - release).total_seconds() / 86400.0)
    horizon_days = case["horizon_minutes"] / 1440.0

    events = _refresh_events(case["history"], n)
    rules = [_classify(t) for t in trophies]
    sens = _sensitivities(events, n, [r[3] for r in rules])
    players_target = _expected_players(case, case["history"], events)
    new_players = max(0.0, players_target - last_players)

    rows = []
    for i, trophy in enumerate(trophies):
        prior, tau, sigma_base, _ = rules[i]
        rate = last_rates[i]
        missing = rate is None
        if missing:
            rate = 0.0
        if rate >= 99.99:
            # Trophy yang sudah 100%: refresh pun tak menurunkannya.
            rows.append({"id": trophy["id"], "q": [99.5, 99.8, 99.95, 99.99, 100.0]})
            continue
        # Kalibrasi: rate terakhir vs yang diprediksi model pada checkpoint.
        # Trophy yang tak terbaca di read terakhir (rate null) tidak
        # dikalibrasi dari nol -- pakai prior apa adanya (cal = 1).
        predicted_check = _matured(prior, tau, days_check)
        if missing or predicted_check <= 0.01:
            cal = 1.0
        else:
            cal = rate / predicted_check
        cal = min(CAL_MAX, max(CAL_MIN, cal))
        # Kohort lama: hanya bisa naik (peraih bertambah), menuju prior.
        old_rate = max(rate, cal * _matured(prior, tau, days_target))
        old_rate = min(99.9, old_rate)
        # Kohort baru: umur efektif sepertiga horizon; trophy yang paling
        # sensitif terhadap refresh dikejar paling lambat.
        new_age = horizon_days / 3.0
        new_rate = cal * _matured(prior, tau, new_age) * (1.0 - 0.35 * sens[i])
        new_rate = min(prior, max(0.0, new_rate))
        median = (last_players * old_rate + new_players * new_rate) / players_target
        median = _clamp_pct(median)
        sigma = sigma_base * math.sqrt(horizon_days + 0.10)
        if not events:
            sigma += UNSEEN_REFRESH_SIGMA
        if median < 0.2:
            # Trophy paling langka: jatuhnya saat refresh paling tak pasti.
            sigma *= 1.15
        sigma = min(2.5, max(0.15, sigma))
        centre = _logit(median)
        quantiles = []
        prev_q = 0.0
        for z in Z:
            q = _round4(_clamp_pct(_inv_logit(centre + z * sigma)))
            q = max(q, prev_q)  # jaga non-decreasing setelah pembulatan
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
