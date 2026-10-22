# PROGRESS.md — kayıt defteri

Her session başında oku. Büyük iş bitince / önemli iş planlanınca en alta (Log) tarihli kayıt ekle. Kararlar `CLAUDE.md`'de.

## Durum
- Faz: kurulum tamam (venv + pinli requirements). Kod henüz yok; sırada feature extractor.
- Repo: https://github.com/emirhannkilic/flappy-bird-rl (boş açıldı)
- Açık bilgiler: grup büyüklüğü (3?), teslim tarihi, GPU/Colab durumu.

## Sıradaki işler
1. ~~venv + requirements + env doğrulaması~~ (bitti, Log'a bak)
2. `src/features.py` (hedef boru seçimi + 4 özellik) + testi.
3. Ayrıklaştırıcı + testi.
4. Q-learning ajanı + train/eval döngüsü, 100k step benchmark.
5. DQN ajanı + testleri.
6. Baseline'lar, 5 seed deneyleri, ablation'lar.
7. Videolar + GitHub Pages sitesi, ardından rapor (LaTeX).

## Log
- 2026-10-22: Proje seçildi (flappy-bird-gymnasium). Rapor şablonu incelendi. Env kaynak kodu doğrulandı (ödül üzerine yazılıyor, terminated=çarpışma, truncated=score_limit). Yöntem seçimi Codex ile tartışıldı: tabular Q-learning vs DQN, aynı 4 özellik. Kararlar `CLAUDE.md`'ye kilitlendi.
- 2026-10-22: `.venv` (Python 3.14.0) + `requirements.txt` (pinli) kuruldu. Env doğrulandı: obs (12,) **int64** (normalize_obs=False; uyarı zararsız, float'a cast et), `max_episode_steps=None`, aynı seed → aynı trajectory, ~264k step/s (render kapalı) → bütçe sorunu yok. Random ajan ~50 step'te ölüyor, çoğu step'te tavan cezası (-0.5) alıyor.
  - Env sabitleri: screen 288x512, player_x=57, PLAYER_WIDTH=34, PIPE_WIDTH=52, pipe_gap=100, ground_y=404.48. Obs'ta boru x'i **sol kenar**; ekrana girmemiş borular sentinel `(288, 0, 512)` olarak geliyor (gerçek boru olarak sayma). Boru geçişi env'de `pipe_mid <= player_mid` anında sayılıyor.
  - **KARAR BEKLİYOR (CLAUDE.md sapması):** hedef boru kuralı "orta noktası kuşun önünde" yerine "sağ kenarı (x+52) kuşun sol kenarını (57) henüz geçmemiş ilk gerçek boru" olmalı. Orta nokta kuralıyla boru ortasını geçer geçmez bir sonraki boşluğa geçilir; kuş hâlâ eski borunun sütunuyla çakışıyor ve çarpabilir.
