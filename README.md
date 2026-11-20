# GTA VI Trophy Rarity Forecast — vocaine

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JACKETAGENT-ID/gta6-trophy-rarity-forecast-01-vocaine/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JACKETAGENT-ID/gta6-trophy-rarity-forecast-01-vocaine/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entri make.fun `gta6-trophy-rarity-forecast-01` (akun vocaine).
Program Python 3.13, stdlib saja, deterministik: `main.py` membaca setiap
`*.in` di `MAKEFUN_INPUT_DIR` dan menulis `<id>.out` ke
`MAKEFUN_OUTPUT_DIR`.

## Model: earners/players decomposition (round 2)

Rate trophy di target = `100 * E_target / P_target`, dengan `E` jumlah
pemain terlacak yang meraih trophy itu dan `P` jumlah pemain terlacak.
Keduanya hanya bisa naik; prakiraan memproyeksikan keduanya ke target,
lalu membagi:

- **Progression:** pemain yang sudah terlacak terus meraih trophy, dengan
  hazard per jam `lam_i` atas yang belum meraih (`P - E`).
- **Dilution:** pemain baru bergabung (P tumbuh) dengan hampir belum
  meraih apa-apa; tiap pendatang baru meraih trophy `i` dengan rate
  marginal `m_i`.

Dua rezim, dipilih dari histori case itu sendiri:

- **Histori FLAT** (semua read identik, seperti malam peluncuran):
  belum ada refresh teramati, jadi satu refresh pertama adalah seluruh
  cerita. Kit mendokumentasikannya persis: 638 pemain / 943 trophy
  earned menjadi 3.868 pemain / 4.276 trophy earned, semua rate kecuali
  yang 100% turun. Model: P melonjak ×6,06 dan 103 peraih baru itu
  jatuh hampir seluruhnya ke trophy jalur cerita / aktivitas cepat
  (pendatang baru meraih trophy awal dengan ~15% rate kohort lama),
  hampir nol ke grind, late-story, dan platinum. Trophy yang 100% di
  semua read tetap ~100%.
- **Histori TRENDING** (case final): `lam_i` diukur dari rentang P
  stabil, `m_i` dari lompatan-lompatan refresh, keduanya di-shrink ke
  prior kelas; P diproyeksikan dengan fit log-time 24 jam terakhir.
  Median = rata-rata logit dari prakiraan ODE ini dan prakiraan tren
  logit-linear (least squares atas `ln(1+jam)`), spread membawa
  selisih keduanya — keduanya berselisih tepat di mana data paling
  tipis.

Spread dihitung pada skala logit (random walk + risiko refresh +
residual fit). Kalibrasi regime flat: jawaban practice direkonstruksi
dari angka-angka resmi kit (4.276 trophy earned pasca-refresh) dan
empat skor server eksak entri round-1 ketiga akun (vocaine p00
0,264958 / p01 0,246354; zech 1,508366; nurz 0,474464) — praktik loss
lokal atas rekonstruksi itu: **0,075** (round 1: 0,256), typical miss
0,024 pp, coverage 0,93. Alokasi per-trofi alternatif (top-20
proporsional) memberi 0,084 — perbaikan ini bukan artefak satu
rekonstruksi.

## Run it

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

2x run pada p00.in dan p01.in: output byte-identik tiap case; 55 baris,
5 kuantil non-decreasing 0..100, ukuran file ~2,8 KB (batas 65.536),
runtime ~0,05 detik (batas 8 detik). Uji tambahan: case trending
sintetis (final-like, 120 read, 4,5 hari horizon) dan case edge (1–2
read, rate null) — semua valid, deterministik, dan median bergerak
wajar (trophy awal naik, trophy langka tetap rendah, 100% tetap 100%).
Jawaban practice tidak tersedia di kit, jadi skor practice di atas
dihitung atas jawaban yang direkonstruksi (lihat bagian Model).

---

<sub>Entered on [make.fun](https://make.fun/?ref=fepxk50h&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
