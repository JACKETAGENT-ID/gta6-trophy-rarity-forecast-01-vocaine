# GTA VI Trophy Rarity Forecast — vocaine

Entri make.fun `gta6-trophy-rarity-forecast-01` (akun vocaine).
Program Python 3.13, stdlib saja, deterministik: `main.py` membaca setiap
`*.in` di `MAKEFUN_INPUT_DIR` dan menulis `<id>.out` ke
`MAKEFUN_OUTPUT_DIR`.

## Model: kurva tier empiris

Rate trophy di target = campuran dua kohort pemain:

- **Kohort lama** (terlacak di checkpoint): ratenya di dalam kohort hanya
  bisa naik; merangkak menuju prior minggu-1 trophy itu.
- **Kohort baru** (bergabung setelah checkpoint): mulai dari nol, mengejar
  prior yang sama dengan umur efektif sepertiga horizon.

Prior minggu-1 ditetapkan dari tier + deskripsi trophy (posisi chapter
untuk story, kata kunci grind/koleksi untuk trophy lambat, platinum
~0,3%), lalu tiap trophy dikalibrasi terhadap rate terakhir yang terbaca.
Jumlah pemain di target: ekstrapolasi laju pertumbuhan yang teramati bila
histori sudah memuat refresh; bila belum ada refresh dan checkpoint masih
dalam 24 jam pertama sejak rilis, diasumsikan satu refresh malam
peluncuran (pemain x6,06, sesuai RULES). Sensitivitas tiap trophy
terhadap gelombang pemain baru dibaca dari refresh teramati di histori.

Spread dihitung pada skala logit, membesar dengan akar horizon dan lebih
lebar untuk trophy langka/grind, plus ketidakpastian refresh bila belum
ada refresh teramati. Trophy yang sudah 100% diprakirakan tetap ~100%.

## Uji lokal

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

2x run pada p00.in dan p01.in: output byte-identik tiap case; 55 baris,
5 kuantil non-decreasing 0..100, ukuran file ~2,8 KB (batas 65.536).
Jawaban practice tidak tersedia di kit, jadi skor practice tidak bisa
dihitung.
