# CLAUDE.md — CMP4501 Flappy Bird RL

**Her session'a başlarken önce `PROGRESS.md` dosyasını oku** (durum, son yapılanlar, sıradaki işler). Büyük bir iş bitince veya önemli bir iş planlanınca `PROGRESS.md`'ye kayıt ekle (tarih + ne + neden/sonuç). Bu dosyadaki kararlar kilitlidir; sapmak gerekirse önce sor, sonra buraya yaz.

Repo: https://github.com/emirhannkilic/flappy-bird-rl

## Kurallar
- Koddaki her şey İngilizce: değişken/fonksiyon adları, yorumlar, docstring'ler, commit mesajı önerileri, README, config anahtarları. Bu dosya ve PROGRESS.md Türkçe.
- `git commit` / `git push` yok; kullanıcı kendisi yapar. Stage + commit mesajı önermek serbest. Commit mesajına `Co-Authored-By: Claude` trailer'ı **asla** ekleme.
- Küçük, test edilebilir adımlar. Production-grade: edge-case ve test kapsamı atlanmaz.
- ML kontrolleri sessizce geçilmez: data leakage (tuning seed'leri ≠ raporlanan seed'ler), seed yönetimi, truncated/terminated ayrımı.
- Rapor (LaTeX, `docs/CMP4501_Report_Template/`) sonraya bırakıldı; şimdilik kod + deney.
- Şablon dosyaları `.gitignore`'da (repoya girmez). `docs/` klasörünün kendisi tracked kalır: GitHub Pages `/docs`'tan yayınlanacak.

## Ortam
- `flappy-bird-gymnasium==0.4.0`, `FlappyBird-v0`, `use_lidar=False`, `normalize_obs=False` (ham piksel, raporda yorumlanabilir).
- Render eğitimde kapalı; sadece final videolarında `rgb_array`.
- Env gerçekleri (kaynak koddan doğrulandı): ödül **üzerine yazılır** (alive +0.1 / boru +1 / tavan -0.5 / ölüm -1); `terminated` sadece çarpışma; `truncated` sadece `score_limit` verilirse; `info["score"]` = geçilen boru sayısı.
- Eğitim: `score_limit=100`. Eval: aynı veya daha yüksek; çoğu ajan tavana ulaşırsa limit artırılır, time-to-cap raporlanır.

## State (iki yöntem için aynı 4 özellik)
Hedef boru = orta noktası kuşun orta noktasının **önünde** olan en yakın boru (borular x'e göre sıralanır; `pipes[0]` kuşun arkasında kalabilir, sabit indeks kullanma).
1. `dx`: hedef boruya yatay uzaklık
2. `dy`: kuş merkezi − boşluk merkezi
3. `vy`: dikey hız
4. `y`: kuşun mutlak dikey konumu (`dy` tek başına tavan/zemin yakınlığını ayıramaz)

## Yöntemler
- **A: tabular Q-learning**, bin: dx 12 × dy 15 × vy 20 × y 10 = 36 000 durum × 2 aksiyon. α=0.1, γ=0.99.
- **B: DQN** (PyTorch, kendi implementasyonumuz; repo'nun TF tabanlı `--mode dqn`'i kullanılmaz). 4→64→64→2 ReLU, Adam 3e-4, batch 64, replay 50k, warm-up 5k, Huber loss, target sync 2k step, grad clip 10, 4 step'te 1 update.
- Exploration (ikisi aynı): ε 1.0→0.05, ilk 300k step lineer.
- Bütçe: 750k–1M env step/seed. Episode sayısıyla durdurma yok.
- Bootstrap maskesi `1 - terminated` (truncated bootstrap'ı kapatmaz).
- Baseline: random; el yazımı (boşluk merkezinin altındaysa ve aşağı gidiyorsa zıpla).
- SARSA ve LIDAR ana deneyde YOK.

## Protokol
- 5 raporlanan seed. Hiperparametre arama seed'leri bunlardan **ayrı**.
- Seed: Python, NumPy, Torch, action_space, env reset (yalnızca run başında; her episode aynı seed ile reset edilmez).
- Eval: greedy (ε=0), ayrı eval env + seed, 100 episode.
- Metrikler: ana = `info["score"]`; ayrıca capped-success rate, episode length, step-to-threshold (şablondaki "episodes to converge" yerine env step; raporda gerekçelendir). Return ile model seçilmez.
- Arama aralıkları: α {0.05, 0.1, 0.2}; DQN lr {1e-4, 3e-4, 1e-3}; γ {0.97, 0.99, 0.995}; ε-decay {100k, 300k, 600k}.
- Önce 100k step benchmark, süreyi doğrusal ölçekle (süre tahminlerine güvenme).

## Ablation (rapor Bölüm 6, tek faktör)
1. Ayrıklaştırma çözünürlüğü: ~4.3k / 36k / ~150k durum (Q-learning).
2. DQN: 4 özellik vs ham 12 özellik (ödül aynı).
3. Yaşama ödülü +0.1 açık/kapalı (diğer olaylar korunur).
4. ε-decay 100k / 300k / 600k.
Reward shaping eklenecekse yalnızca potansiyel tabanlı F = γΦ(s′) − Φ(s).

## Zorunlu testler
- Hedef boru seçimi: boru geçişinden önce/sonra.
- Ayrıklaştırıcı: sınır değerler, bin sayısı, determinizm.
- DQN target maskesi: truncated vs terminated.
- Seed tekrarlanabilirliği: aynı seed → aynı eval skoru.

## Repo yapısı
`src/` (features, agents, train, eval), `configs/`, `results/`, `docs/` (rapor + Pages), `tests/`, `CLAUDE.md`, `PROGRESS.md`, `requirements.txt` (sürümler sabit).

## Ders gereksinimleri (şablondan)
≥2 yöntem, ≥5 seed (ortalama ± std), random baseline, 6–8 sayfa ana metin, GitHub Pages sitesi (ajan videoları), Ek B AI kullanım beyanı zorunlu.
