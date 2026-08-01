# BTC/ETH olasılık araştırma botu

Bu proje yalnızca `BTCUSDT` ve `ETHUSDT` için Binance Spot'un kapanmış `5m`, `15m`
ve `1h` mumlarını kullanır. Bir sonraki mum kapanışının yukarı/aşağı olasılığını
hesaplar; geçmiş walk-forward araştırma kapısını geçen ve en az `%60` kalibre yön
olasılığı üreten tahminleri üçüncü Telegram kanalına yollar. Emir vermez, borsa hesabına
bağlanmaz ve Binance API anahtarı kullanmaz.

## Tahminin içeriği

Her bildirim şunları gösterir:

- sinyal ve hedef mum zamanı (UTC), son kapanmış mum fiyatı;
- yukarı/aşağı kalibre olasılığı;
- mevcut ATR'nin `+0.5` ve `-0.5` katındaki fiyatlara bir sonraki mum içinde dokunma
  olasılığı;
- benzer geçmiş senaryolardan `%80` kapanış aralığı ve medyan kapanış;
- yalnızca kronolojik test dilimlerinden ölçülen yüksek güven başarı oranı, `%95 Wilson`
  güven aralığı, örnek sayısı, kapsama, Brier ve ECE;
- tahmini en çok etkileyen dört belirteç ve yönü destekleyip zayıflatmaları.

Bu fiyat senaryoları “fiyat kesin buraya gelir” demek değildir. Olasılık kovaları ile
oynaklık rejiminde bulunan geçmiş benzer örneklerin ampirik frekanslarıdır.

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
kullanılmaz. Her sembol/zaman dilimi ayrı düzenlileştirilmiş lojistik modeldir; amaç
yorumlanabilir ve kalibre edilebilir bir taban oluşturmaktır.

## Araştırma protokolü

Veri rastgele karıştırılmaz. Her fold şu sıradadır:

`geçmiş train → 1 hedef mum embargo → daha yeni kalibrasyon → 1 hedef mum embargo → daha yeni test`

Model train bölümünde öğrenir; Platt olasılık kalibrasyonu daha sonraki kalibrasyon
bölümünde yapılır; raporlanan bütün sonuçlar ikisinden de sonraki test bölümündendir.
Fold ilerledikçe zaman ileri akar. Bildirim kapısı şu koşulların tamamını ister:

- en az 100 yüksek güven OOS örneği;
- yüksek güven yön doğruluğu en az `%53`;
- altı model birlikte değerlendirildiği için Bonferroni aile-düzeltmeli `%95 Wilson`
  alt sınırı `%50` üzerinde;
- Brier skoru fold'un yalnız tarihsel yukarı oranını kullanan tabandan iyi;
- ECE en fazla `%10`.

Bir model kapıda kalırsa veya güncel olasılık `%60` altında kalırsa Telegram mesajı
gönderilmez. Başarı oranı geleceğe dair garanti değildir; piyasa rejimi değişebilir.

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

Tek seferlik gerçek değerlendirme ve yalnız uygunsa gönderim:

```powershell
python run.py predict --refresh --send
```

Sürekli çalışma; veriyi dakikada bir yeniler, modelleri UTC gününde bir kez yeniden
araştırır ve aynı mum/sinyali ikinci kez göndermez:

```powershell
python run.py serve --days 365 --poll-seconds 60
```

Windows Görev Zamanlayıcı veya bir servis yöneticisiyle bu komut makine açılışında
başlatılabilir. `serve` kesintisiz çalışır; işlem kapalıyken bildirim oluşmaz.

## Tamamen bulutta çalışma

`.github/workflows/cloud-bot.yml` botu GitHub Actions üzerinde beş dakikada bir
çalıştırır. Mum/veri/model durumu Actions cache içinde korunur; walk-forward araştırma
yaklaşık günde bir yenilenir. Uygun sinyal oluşursa Telegram mesajı gönderilir ve
Serhan / Lab panelinin güvenli veri kapısına son durum aktarılır.

GitHub deposunda şu Actions secrets tanımlanmalıdır:

- `CRYPTO_TELEGRAM_BOT_TOKEN`
- `CRYPTO_TELEGRAM_CHAT_ID`
- `PROJECT_HUB_INGEST_URL`
- `PROJECT_HUB_INGEST_TOKEN`
- özel Sites yayını kullanılıyorsa `OAI_SITES_BYPASS_TOKEN`

Bulut çalışması elle de başlatılabilir; `force_research` seçeneği bütün modelleri
yeniden araştırır. Üçüncü Telegram kanalı bağlandıktan sonra `telegram_test` seçeneği
kanala tek bir bağlantı test mesajı yollar. Token değerlerini repoya veya workflow
dosyasına yazmayın.

## Sınırlar

- Veri kaynağı tek borsadır (Binance Spot); başka borsadaki fiyat/likidite farklı olabilir.
- Backtest yön tahminini ölçer; komisyon, kayma ve emir gerçekleşmesini ölçmez.
- Çok sayıda model/parametre denemesi yapılmaz; yine de altı model birlikte incelendiği
  için tek bir iyi sonuca aşırı anlam yüklenmemelidir.
- `%50` yakınındaki kısa vadeli piyasa yönü normaldir. Araştırma kapısını hiçbir model
  geçmezse doğru davranış bildirim göndermemektir.
- Bu yazılım yatırım tavsiyesi değildir ve otomatik alım-satım yapmaz.

Veri istemcisi Binance'in resmî salt-okunur market-data kökünü, bildirim istemcisi
Telegram Bot API `sendMessage` metodunu kullanır.

Resmî teknik dayanaklar: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
ve [Telegram Bot API sendMessage](https://core.telegram.org/bots/api#sendmessage).
