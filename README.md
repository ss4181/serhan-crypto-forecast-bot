# BTC/ETH olasılık araştırma botu

Bu proje yalnızca `BTCUSDT` ve `ETHUSDT` için Binance Spot'un kapanmış `5m`, `15m`
ve `1h` mumlarını kullanır. Bir sonraki mum kapanışının yukarı/aşağı olasılığını
hesaplar ve sonucu Telegram kanalına yollar. Emir vermez, borsa hesabına bağlanmaz
ve Binance API anahtarı kullanmaz.

## İki bildirim seviyesi

Yön isabetinin `%50` üzerinde olması para kazandırmaz: kazanan ve kaybeden
işlemlerin büyüklüğü farklıdır ve her gidiş-dönüş komisyon öder. Bu yüzden
bildirimler iki seviyeye ayrılmıştır.

| Seviye | Ne demek | Ne zaman gelir |
|---|---|---|
| **İŞLEM ADAYI** | Modelin maliyet sonrası ölçülen beklentisi pozitif ve bu beklentinin gün bloklu `%95` alt sınırı sıfırın üstünde | Yalnız araştırma kapısını geçen modellerden, hedef mumun en az `%60`'ı önündeyken |
| **GÖZLEM** | Model bir tahmin üretti ama işlem beklentisi kanıtlanmadı | Altı modelin tamamı için, varsayılan olarak `6` saatte bir tek özet mesaj |

GÖZLEM seviyesi sayesinde **hiçbir model sessiz kalmaz**: kanalda altı modelin de
durumu düzenli olarak görünür, ama beklentisi negatif bir tahmin işlem sinyali
gibi sunulmaz.

**Bugünkü durum:** altı modelin hiçbiri işlem kapısını geçmiyor. En iyi model
(`BTCUSDT 15m`, `%63.4` yön isabeti) bile `20` baz puan gidiş-dönüş maliyeti
düşüldükten sonra sinyal başına `-14.7` bps üretiyor ve bunun `%95` aralığı
tamamen sıfırın altında. Kanal bu nedenle şu an yalnız GÖZLEM raporu yayınlar;
doğru davranış budur.

## Tahminin içeriği

Her bildirim şunları gösterir:

- seviye (İŞLEM ADAYI / GÖZLEM), sinyal ve hedef mum zamanı (UTC), hedef mumda
  kalan süre, son kapanmış mum fiyatı;
- yukarı/aşağı kalibre olasılığı;
- **maliyet sonrası beklenti**: sinyal başına net baz puan, gün bloklu `%95`
  aralığı, ortalama kazanç ve ortalama kayıp;
- mevcut ATR'nin `+0.5` ve `-0.5` katındaki fiyatlara dokunma olasılığı ve
  **ikisinin aynı mumda birlikte görülme olasılığı** — mum verisinden hangisinin
  önce geldiği bilinemez, bu yüzden bu değerler bir hedef/stop çifti değildir;
- benzer geçmiş senaryolardan `%80` kapanış aralığı ve medyan kapanış;
- yalnızca kronolojik test dilimlerinden ölçülen yön isabeti, örnek sayısı, kaç
  ayrı güne dağıldığı, kapsama, Brier ve ECE;
- tahmini en çok etkileyen dört belirteç ve yönü destekleyip zayıflatmaları.

## Seçilen belirteçler

Model yalnızca mumun o anda bilinen OHLCV değerlerinden şu dokuz girdiyi üretir:

1. ATR'ye bölünmüş 1, 3 ve 12 mum log momentumu;
2. ATR'ye bölünmüş EMA(8)–EMA(21) farkı;
3. RSI(14);
4. Bollinger(20) z-konumu;
5. log ATR(14) yüzdesi (oynaklık rejimi);
6. 20 mumluk log-hacim z-skoru;
7. kapanışın mum aralığındaki alıcı/satıcı baskısı.

Haber, sosyal medya, zincir verisi, fonlama, emir defteri veya gelecekte oluşan bir değer
kullanılmaz. Her sembol/zaman dilimi ayrı düzenlileştirilmiş lojistik modeldir.

## Araştırma protokolü

Veri rastgele karıştırılmaz. Her fold şu sıradadır:

`geçmiş train → 1 hedef mum embargo → daha yeni kalibrasyon → 1 hedef mum embargo → daha yeni test`

Model train bölümünde öğrenir; Platt olasılık kalibrasyonu daha sonraki kalibrasyon
bölümünde yapılır; raporlanan bütün sonuçlar ikisinden de sonraki test bölümündendir.
Bildirim kapısı şu koşulların tamamını ister:

- en az 100 yüksek güven OOS örneği;
- yüksek güven yön doğruluğu en az `%53`;
- altı model birlikte değerlendirildiği için Bonferroni aile-düzeltmeli `%95 Wilson`
  alt sınırı `%50` üzerinde;
- Brier skoru fold'un yalnız tarihsel yukarı oranını kullanan tabandan iyi;
- ECE en fazla `%10`;
- **gidiş-dönüş maliyeti düşüldükten sonra sinyal başına beklenti pozitif**;
- **bu beklentinin gün bloklu bootstrap ile hesaplanan `%95` alt sınırı sıfırın
  üstünde.**

Son iki koşul belirleyicidir. Yüksek güven sinyalleri aynı seansta kümelendiği
için bağımsız örnek varsayan güven aralığı fazla dardır; bootstrap tüm günü
birlikte yeniden örnekler.

Maliyet `Settings.round_trip_cost_bps` ile ayarlanır. Varsayılan `20.0`, Binance
Spot taker ücretinin (`%0.10` × 2) karşılığıdır. Vadeli taker için `10.0` yazın.
Kayma ve emir gerçekleşmesi hâlâ ölçülmez, yani gerçek maliyet bundan yüksektir.

## Kurulum (Windows PowerShell)

Python 3.11+ kurulu olmalıdır:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

İlk araştırma yaklaşık bir yıllık veriyi indirir, altı modeli test eder ve kaydeder:

```powershell
python run.py research --days 365
```

Çıktılar:

- `data/BTCUSDT_5m.csv` benzeri yerel kapalı mum önbellekleri;
- `artifacts/models/*.json` model/kalibrasyon/backtest paketleri;
- `artifacts/reports/latest_backtest.md` okunabilir karşılaştırma;
- `artifacts/reports/latest_backtest.json` makinece okunabilir tam metrikler.

Mevcut veriye ağsız tekrar test:

```powershell
python run.py research --offline
```

Telegram'a göndermeden son tahminleri görüntüleme:

```powershell
python run.py predict --refresh
```

Testler:

```powershell
python -m unittest discover -s tests -t tests
```

## Üçüncü Telegram kanalı

Telegram'da üçüncü kanalı oluşturun. BotFather ile yeni bir bot oluşturabilir veya mevcut
bildirim botunu bu kanala ekleyebilirsiniz. Botu kanal yöneticisi yapıp mesaj gönderme
yetkisi verin. Token'ı repoya ya da komut satırı argümanına yazmayın.

Bot token'ı ve kanal kimliğini kullanıcı ortamına kaydedin:

```powershell
[Environment]::SetEnvironmentVariable(
  "CRYPTO_TELEGRAM_BOT_TOKEN",
  "BOTFATHER_TOKENINIZ",
  "User"
)
[Environment]::SetEnvironmentVariable(
  "CRYPTO_TELEGRAM_CHAT_ID",
  "@kanal_kullanici_adi_veya_-100_ile_baslayan_id",
  "User"
)
```

Yeni bir terminal açıp bağlantıyı sınayın:

```powershell
python run.py telegram-test
```

### Altı modelin de kanala ulaştığını doğrulama

Sessiz bir kanal iki anlama gelebilir: hiçbir model kapıyı geçmemiştir ya da
boru hattı bozulmuştur. Bu komut ikisini ayırır — araştırma kapısından bağımsız
olarak her modelin mesaj üretip üretemediğini tek tek dener:

```powershell
python run.py verify-models --refresh
```

`--send` eklendiğinde altı modelin her biri için kanala birer doğrulama mesajı
gider, yani kanalda hepsini gözle görürsünüz:

```powershell
python run.py verify-models --refresh --send
```

Bulutta aynı doğrulama, workflow'u elle çalıştırırken `verify_models` seçeneğiyle
yapılır.

### Canlı karne

Gönderilen her sinyal, hedef mumu kapandığında gerçek sonucuyla eşleştirilip
`state/outcomes/ledger.jsonl` dosyasına yazılır. Backtest modeli ölçer, karne
botun kendisini ölçer:

```powershell
python run.py scorecard --days 30
```

Bulut çalışması bu özeti UTC gününde bir kez kanala da yollar.

### Tek seferlik gerçek değerlendirme

```powershell
python run.py predict --refresh --send
```

Sürekli çalışma; veriyi dakikada bir yeniler, modelleri UTC gününde bir kez yeniden
araştırır ve aynı mum/sinyali ikinci kez göndermez:

```powershell
python run.py serve --days 365 --poll-seconds 60
```

`serve` ağ hatasında ölmez; artan bekleme ile yeniden dener.

## Tamamen bulutta çalışma

`.github/workflows/cloud-bot.yml` botu GitHub Actions üzerinde beş dakikada bir
çalıştırır. Walk-forward araştırma yaklaşık günde bir yenilenir.

Durum iki ayrı Actions cache'inde tutulur:

- `data` (mumlar) saatlik anahtarla — büyük ama kendi kendini onarır, kaybı birkaç
  isteğe mal olur;
- `artifacts` + `state` (model paketleri ve teslimat makbuzları) **her koşuya özel**
  anahtarla. `actions/cache` birebir anahtar isabetinde kaydetme adımını atladığı
  için ortak saatlik anahtar, saatin ilk koşusu dışındaki bütün makbuzları çöpe
  atıyordu; bu da aynı sinyalin tekrar tekrar gönderilmesine yol açıyordu.

GitHub deposunda şu Actions secrets tanımlanmalıdır:

- `CRYPTO_TELEGRAM_BOT_TOKEN`
- `CRYPTO_TELEGRAM_CHAT_ID`
- `PROJECT_HUB_INGEST_URL` — panelin **`/api/ingest`** yolu olmalıdır;
  `/api/project-ingest` farklı bir şemayı doğrular ve gönderimi `400` ile reddeder
- `PROJECT_HUB_INGEST_TOKEN`
- özel Sites yayını kullanılıyorsa `OAI_SITES_BYPASS_TOKEN`

Bulut çalışması elle de başlatılabilir; `force_research` bütün modelleri yeniden
araştırır, `telegram_test` tek bir bağlantı testi yollar, `verify_models` altı
modelin de kanala ulaştığını doğrular. Token değerlerini repoya veya workflow
dosyasına yazmayın.

## Sınırlar

- Veri kaynağı tek borsadır (Binance Spot); başka borsadaki fiyat/likidite farklı olabilir.
- Backtest komisyonu hesaba katar ama kaymayı ve emir gerçekleşmesini ölçmez.
- GitHub'ın `*/5` cron'u yoğunlukta gecikir. Tazelik kapısı bu yüzden hedef mumda
  kalan süreye bakar; geciken bir koşu `5m` sinyalini göndermek yerine düşürür.
  Kısa vadeli modellerin bulutta seyrek tetiklenmesi beklenen davranıştır.
- Çok sayıda model/parametre denemesi yapılmaz; yine de altı model birlikte incelendiği
  için tek bir iyi sonuca aşırı anlam yüklenmemelidir.
- `%50` yakınındaki kısa vadeli piyasa yönü normaldir. Araştırma kapısını hiçbir model
  geçmezse doğru davranış işlem sinyali göndermemektir.
- Bu yazılım yatırım tavsiyesi değildir ve otomatik alım-satım yapmaz.

Veri istemcisi Binance'in resmî salt-okunur market-data kökünü, bildirim istemcisi
Telegram Bot API `sendMessage` metodunu kullanır.

Resmî teknik dayanaklar: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
ve [Telegram Bot API sendMessage](https://core.telegram.org/bots/api#sendmessage).
