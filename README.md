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
| **GÖZLEM** | Model bir tahmin üretti ama işlem beklentisi kanıtlanmadı | Altı modelin tamamı için **günde bir** tek özet mesaj |

GÖZLEM seviyesi sayesinde **hiçbir model sessiz kalmaz**: kanalda altı modelin de
durumu günlük olarak görünür, ama beklentisi negatif bir tahmin işlem sinyali
gibi sunulmaz. Aradaki zamanda gereksiz mesaj gelmez; merak ettiğinizde
`/durum` yazarak anlık cevabı alırsınız.

**Bugünkü durum:** üçlü bariyer hedefi ve sızıntısız embargo ile altı modelin
hiçbiri işlem kapısını geçmiyor. Anlamlı örneğe sahip tek model `BTCUSDT 1h`
(216 sinyal, 51 ayrı gün): yön isabeti `%45.4` ve maliyet sonrası beklentisi
sinyal başına `-24.7` bps, `%95` aralığı tamamen negatif. Kanal bu nedenle şu an
yalnız GÖZLEM raporu yayınlar; doğru davranış budur.

## Hedef: üçlü bariyer

Eski hedef "bir sonraki mumun kapanışı daha yüksek mi" idi. Bu hedef bir baz
puanlık sürüklenme ile gerçek bir hareketi aynı olay sayar, bu yüzden bir model
komisyonlara para kaptırırken isabetli görünebilir.

Yeni hedef, o mumda açılan bir işlemin **önce hangi bariyere** değdiğidir:

- **Kâr al / zarar kes**: girişten `±X` baz puan uzaklıkta simetrik iki seviye.
- **Süre bariyeri**: en fazla `barrier_horizon_candles` mum (varsayılan `12`);
  o zamana kadar bir tarafa değilmezse işlem piyasadan kapanır.
- **`X` asla maliyetin altına inmez**: `barrier_cost_multiple` (varsayılan `2`)
  ile gidiş-dönüş maliyetinin en az iki katı olarak taban alınır. Yani kazanan
  bir işlem her zaman kendi masrafını fazlasıyla karşılar.
- **Aynı mumda iki seviye birden görülürse** hangisinin önce geldiği mum
  verisinden bilinemez; bu durum **her iki yön için de zarar** sayılır. Bir
  tacir kanıtlayamadığı iyi sonucu varsayamaz.

Bariyer etiketi birden fazla mum ileriye baktığı için train/kalibrasyon/test
dilimleri arasındaki embargo **tam bir etiket ufku** kadardır. Tek mumluk
embargo, eğitim setinin komşusunun puanlandığı fiyat hareketini görmesine izin
verirdi.

## Tahminin içeriği

Her bildirim şunları gösterir:

- seviye (İŞLEM ADAYI / GÖZLEM), sinyal ve hedef mum zamanı (UTC), hedef mumda
  kalan süre, son kapanmış mum fiyatı;
- hedef tanımı: bariyer uzaklığı, süre sınırı ve geçmişte sinyallerin yüzde
  kaçının bariyere ulaştığı;
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

`geçmiş train → etiket ufku kadar embargo → daha yeni kalibrasyon → etiket ufku kadar embargo → daha yeni test`

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

### Bota soru sorma

Bot yalnız mesaj göndermez; kendisine sorulduğunda cevap verir. Bunun için
sahibin sayısal Telegram kimliği gerekir (`@userinfobot` gibi bir bota yazarak
öğrenebilirsiniz):

```powershell
[Environment]::SetEnvironmentVariable("CRYPTO_TELEGRAM_OWNER_ID", "123456789", "User")
```

Sonra bota **özel mesaj** olarak:

| Komut | Ne yapar |
|---|---|
| `/durum` | Altı modelin o anki durumu — beklemeden anlık cevap |
| `/performans [gün]` | Gönderilen sinyallerin gerçek sonucu (varsayılan 30 gün) |
| `/kisiler` | Yetkili kişiler |
| `/yardim` | Komut listesi |
| `/ekle <kimlik> <ad>` | **Yalnız sahip** — birine sorgulama yetkisi verir |
| `/sil <kimlik>` | **Yalnız sahip** — yetkiyi kaldırır |

Yetki kuralları:

- `CRYPTO_TELEGRAM_OWNER_ID` tanımlı değilse komut yanıtlama tamamen kapalıdır.
- Listede olmayan birinin komutu **sessizce yok sayılır**. Cevap verilseydi
  herkes botu istediği anda mesaj üretmeye zorlayabilirdi.
- Eklenen kişi yalnızca **sorgulama** yetkisi alır. Hiçbir komut emir veremez,
  pozisyon açamaz veya para hareketi başlatamaz; botun böyle bir yüzeyi yoktur.
- Gelen mesaj metni sabit bir komut tablosuyla eşleştirilir; hiçbir zaman
  talimat olarak yorumlanmaz. Kişi adları da yazdırılmadan önce temizlenir.

Kişileri komut satırından da yönetebilirsiniz:

```powershell
python run.py members --add 123456789 --name "Ayse Yilmaz"
python run.py members --remove 123456789
python run.py members
```

Bulut çalışması her turda bekleyen komutları okuyup yanıtlar. Yerelde tek sefer
denemek için:

```powershell
python run.py commands
```

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
- `CRYPTO_TELEGRAM_OWNER_ID` — komutlara cevap verilmesi için gerekir
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
- **GitHub `*/5` cron'unu fiilen uygulamıyor.** Ölçülen: bu depoda üç saatte
  beklenen ~35 koşu yerine 2 zamanlanmış koşu çalıştı, yani gerçek tempo saatte
  bir. Tazelik kapısı hedef mumda kalan süreye baktığı için geciken bir koşu
  `5m`/`15m` sinyalini göndermek yerine düşürür. Bulutta pratikte yalnız `1h`
  modelinin tetiklenme penceresi (24 dakika) bu tempoya uyar. Dakika hassasiyeti
  gerekiyorsa `serve` komutunu sürekli açık bir makinede çalıştırın.
- Çok sayıda model/parametre denemesi yapılmaz; yine de altı model birlikte incelendiği
  için tek bir iyi sonuca aşırı anlam yüklenmemelidir.
- `%50` yakınındaki kısa vadeli piyasa yönü normaldir. Araştırma kapısını hiçbir model
  geçmezse doğru davranış işlem sinyali göndermemektir.
- Bu yazılım yatırım tavsiyesi değildir ve otomatik alım-satım yapmaz.

Veri istemcisi Binance'in resmî salt-okunur market-data kökünü, bildirim istemcisi
Telegram Bot API `sendMessage` metodunu kullanır.

Resmî teknik dayanaklar: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
ve [Telegram Bot API sendMessage](https://core.telegram.org/bots/api#sendmessage).
