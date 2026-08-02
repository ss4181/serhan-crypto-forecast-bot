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
| **İŞLEM ADAYI** | Modelin maliyet sonrası ölçülen beklentisi pozitif ve bu beklentinin blok-bootstrap `%95` alt sınırı sıfırın üstünde | Yalnız araştırma kapısını geçen modellerden, hedef mumun en az `%60`'ı önündeyken |
| **GÖZLEM** | Model bir tahmin üretti ama işlem beklentisi kanıtlanmadı | Altı modelin tamamı için **günde bir** tek özet mesaj |

GÖZLEM seviyesi sayesinde **hiçbir model sessiz kalmaz**: kanalda altı modelin de
durumu günlük olarak görünür, ama beklentisi negatif bir tahmin işlem sinyali
gibi sunulmaz. Aradaki zamanda gereksiz mesaj gelmez; merak ettiğinizde
`/durum` yazarak anlık cevabı alırsınız.

**Bugünkü durum:** altı modelin hiçbiri işlem kapısını geçmiyor. En iyisi
`BTCUSDT 5m` (3.333 sinyal, 46 ayrı gün): `%53.9` isabet, brüt `+5.1` bps,
`10` bps maliyetten sonra `-4.9` bps. Gereken `%55`.

Daha rahatsız edici olan, aynı satırdaki **taban** değeri: bariyer hedefinde
yukarı tarafın önce görülme oranı `%52.5`, yani hiç düşünmeden "yukarı" diyen
bir kural modele yakın isabet veriyor. Mevcut dokuz OHLCV belirteci bu hedefte
güçlü bir yön bilgisi taşımıyor. Kanal bu nedenle şu an yalnız GÖZLEM raporu
yayınlar; doğru davranış budur.

## Hedef: üçlü bariyer

Eski hedef "bir sonraki mumun kapanışı daha yüksek mi" idi. Bu hedef bir baz
puanlık sürüklenme ile gerçek bir hareketi aynı olay sayar, bu yüzden bir model
komisyonlara para kaptırırken isabetli görünebilir.

Yeni hedef, o mumda açılan bir işlemin **önce hangi bariyere** değdiğidir:

- **Kâr al / zarar kes**: girişten `barrier_target_bps` kadar uzaklıkta simetrik
  iki seviye. Varsayılan `100` bps, yani hedeflenen `%1`'lik işlem.
- **Süre bariyeri**: en fazla `barrier_horizon_hours` (varsayılan `24` saat);
  o zamana kadar bir tarafa değilmezse işlem piyasadan kapanır. Saat cinsinden
  tanımlıdır, böylece her zaman dilimi aynı piyasa süresini kapsar.
- **Aynı mumda iki seviye birden görülürse** hangisinin önce geldiği mum
  verisinden bilinemez; bu durum **her iki yön için de zarar** sayılır. Bir
  tacir kanıtlayamadığı iyi sonucu varsayamaz.

Bariyer genişliğinin maliyete oranı, işin yapılabilir olup olmadığını tek başına
belirler. Simetrik bariyerde başabaş isabet oranı:

**gerekli isabet = %50 + maliyet / (2 × bariyer)**

- `40` bps bariyer, `20` bps spot maliyet → **%75** gerekir. Pratikte imkânsız.
- `100` bps bariyer, `10` bps vadeli maliyet → **%55** gerekir. Ulaşılabilir bir bar.

Bu yüzden varsayılan maliyet Binance USD-M vadeli taker (`%0.05` × 2 = `10` bps)
ve bariyer `%1`'dir. Spot işlem yapacaksanız `round_trip_cost_bps` değerini `20`
yapın — o zaman bariyeri de genişletmeniz gerekir, yoksa bar yükselir.

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
- **maliyet sonrası beklenti**: sinyal başına net baz puan, blok-bootstrap `%95`
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

Haber, sosyal medya, zincir verisi, fonlama veya gelecekte oluşan bir değer
kullanılmaz. Her sembol/zaman dilimi ayrı düzenlileştirilmiş lojistik modeldir.

### Denenip elenenler

Onbellek Binance'in gönderdiği `quote_volume`, `trade_count` ve
`taker_buy_base` alanlarını da saklar, ama model bunları **kullanmaz**. Emir
akışından türetilen dört ayrı form denendi — taker alıcı/satıcı dengesinin ham
değeri, 12 mumluk ortalaması, 20 mumluk z-skoru ve 3 mumluk değişimi, ayrıca
ortalama işlem büyüklüğü. Aynı walk-forward dilimlerinde 18 model
karşılaştırmasının 16'sında net beklentiyi **kötüleştirdiler**: agresör ayrımı
büyük ölçüde `candle_pressure`'ın tekrarı olduğu için bilgi katmadan varyans
ekliyor. Sütunlar farklı bir model sınıfı tekrar denesin diye tutuluyor.

**Gradient boosting (HistGradientBoosting)** de aynı dilimlerde denendi. Ham
ortalamalar cazip görünüyordu — `ETHUSDT 5m` için `%70.6` isabet ve `+27.3` bps —
ama sinyaller yalnız 10 ayrı haftaya düşüyordu ve blok bootstrap alt sınırı
`-16.0` bps çıktı. Altı modelin hiçbirinde alt sınır sıfırı geçmedi; üstelik
`BTCUSDT 15m`'de isabet `%34.9`'a düştü. Bu tutarsızlık sinyal değil varyans
işaretidir, o yüzden `scikit-learn` bağımlılığı eklenmedi. Bu deney, blok
uzunluğunun etiket ufkundan büyük olması gerektiğini de ortaya çıkardı.

**Vadeli funding oranı** üç formda denendi: son ödenen oran, 30 ödemelik
z-skoru ve son 24 saatin toplamı. Hizalama `merge_asof(direction="backward")`
ile yapıldı, yani bir mum yalnız kapanışından önce ödenmiş funding'i görür.
Altı modelin beşinde net beklenti kötüleşti (en kötüsü `-17.4` bps); tek
iyileşen `BTCUSDT 1h`'de bile alt sınır `-34.8` bps'de kaldı.

**Açık pozisyon (open interest) test edilemedi.** Binance'in
`futures/data/openInterestHist` uç noktası yalnız son ~30 günü tutuyor; 60 gün
öncesi `HTTP 400` veriyor. Bir yıllık walk-forward için yeterli geçmiş yok.
İleride ölçebilmek adına şimdiden günlük kayıt toplamak gerekir.

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
- **bu beklentinin blok bootstrap ile hesaplanan `%95` alt sınırı sıfırın
  üstünde.**

Son iki koşul belirleyicidir. Yüksek güven sinyalleri aynı seansta kümelendiği
için bağımsız örnek varsayan güven aralığı fazla dardır; bootstrap blokları
birlikte yeniden örnekler.

**Blok uzunluğu etiket ufkundan büyük olmalıdır.** Bariyer etiketi ileriye
`barrier_horizon_hours` kadar baktığı için, birbirine bu süreden yakın iki
sinyal aynı fiyat hareketiyle puanlanır. 24 saatlik ufukta takvim günleri bile
örtüşür; blok en az etiket penceresinin iki katı ve asla bir günden kısa
olmayacak şekilde veriden türetilir.

Maliyet `Settings.round_trip_cost_bps` ile ayarlanır. Varsayılan `10.0`, Binance
USD-M vadeli taker ücretinin (`%0.05` × 2) karşılığıdır. Spot için `20.0` yazın.
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

## Dakika hassasiyeti: sürekli açık sunucu

GitHub Actions dakika hassasiyeti için uygun değildir. İki ayrı duvar var:

- **Zamanlama**: GitHub `*/5` cron'unun çoğunu düşürür. Bu depoda ölçüldü — üç
  saatte beklenen ~35 koşu yerine 2 zamanlanmış koşu çalıştı, fiili tempo saatte
  bir.
- **Kota**: depo private olduğu için aylık 2.000 Actions dakikası var ve her iş
  bir sonraki tam dakikaya yuvarlanarak faturalanır. Dakikada bir koşu ayda
  ~43.000 dakika eder. Ayrıca 7/24 döngü, Actions'ı genel amaçlı barındırma
  olarak kullanmayı yasaklayan kullanım şartlarına aykırıdır.

Bu yüzden dakika hassasiyeti sürekli açık bir makinede `serve` ile sağlanır.
`serve` artık `cloud-run` ile aynı işleri yapar: veri yenileme, sonuç
sonuçlandırma, günlük araştırma, bildirim, gözlem raporu, komut yanıtlama ve
panel güncelleme.

### Rol: tek gönderici

`CRYPTO_BOT_ROLE` iki değer alır:

| Rol | Ne yapar |
|---|---|
| `primary` (varsayılan) | Telegram'a mesaj gönderir ve komutları yanıtlar |
| `standby` | Yalnız araştırma yapar ve paneli günceller; **hiç mesaj göndermez** |

Aynı anda iki `primary` kopya çalışırsa her uyarı iki kez gider (teslimat
makbuzları ayrı diskte tutulur) ve Telegram eş zamanlı `getUpdates` çağrılarını
`409` ile reddettiği için komutlar kaybolur. Bu yüzden sunucu `primary`,
GitHub Actions kopyası `standby` olmalıdır — workflow varsayılan olarak
`standby` gelir.

### Oracle Cloud Always Free kurulumu

Always Free ARM makine (4 çekirdek / 24 GB) kalıcı olarak ücretsizdir ve bu iş
yükü için fazlasıyla yeterlidir.

1. Oracle Cloud hesabı açın, **Always Free** etiketli bir **Ampere (ARM)**
   compute instance oluşturun ve SSH anahtarınızı ekleyin. Gelen bağlantıya
   ihtiyaç yok; bot yalnızca dışa doğru HTTPS konuşur, ek port açmayın.

   İmaj olarak Oracle Linux (varsayılan) veya Ubuntu seçebilirsiniz; kurulum
   script'i ikisini de destekler. **SSH kullanıcı adı imaja göre değişir:**
   Oracle Linux'ta `opc`, Ubuntu'da `ubuntu`.
2. Makineye bağlanıp depoyu çekin. Depo private olduğu için sunucuda bir SSH
   anahtarı üretip GitHub'da **Deploy key** (salt okunur) olarak ekleyin.
   Oracle Linux'un minimal imajında `git` kurulu gelmez, önce onu kurun:

   ```bash
   sudo dnf install -y git        # Ubuntu'da: sudo apt-get install -y git
   ssh-keygen -t ed25519 -C "oracle-bot" -f ~/.ssh/id_ed25519 -N ""
   cat ~/.ssh/id_ed25519.pub   # GitHub > Settings > Deploy keys > Add
   git clone git@github.com:ss4181/serhan-crypto-forecast-bot.git
   ```

3. Kurulum:

   ```bash
   cd serhan-crypto-forecast-bot
   sudo bash deploy/install.sh
   ```

4. Gizli değerleri girin ve servisi başlatın:

   ```bash
   sudo nano /etc/crypto-forecaster.env
   sudo systemctl restart crypto-forecaster
   sudo journalctl -u crypto-forecaster -f
   ```

   İlk tur bir yıllık veriyi indirip altı modeli araştırır (5–10 dakika).

5. GitHub tarafında `CRYPTO_BOT_ROLE` repository variable'ını **tanımlamayın**;
   workflow zaten `standby` varsayar. Sunucu devre dışı kalırsa bu değişkeni
   `primary` yaparak Actions'ı geçici gönderici hâline getirebilirsiniz.

Servis `Restart=always` ile çalışır, systemd sertleştirmesi altındadır
(salt okunur kök dosya sistemi, capability yok, yalnızca kendi veri dizinlerine
yazar) ve gizli değerler yalnızca `0600` izinli `/etc/crypto-forecaster.env`
dosyasında durur.

Kod güncellemesi:

```bash
cd ~/serhan-crypto-forecast-bot && git pull
sudo bash deploy/update.sh
```

`update.sh` önce testleri çalıştırır; testler geçmezse güncellemeyi durdurur.

Konteyner tercih ederseniz kökteki `Dockerfile` aynı işi yapar; `data`,
`artifacts` ve `state` dizinlerini kalıcı volume olarak bağlayın — `state`
kaybolursa gönderilmiş bir uyarı ikinci kez gidebilir.

## Yedek olarak bulutta çalışma

`.github/workflows/cloud-bot.yml` üç saatte bir çalışır ve **varsayılan olarak
`standby`** rolündedir: araştırmayı ve paneli güncel tutar, mesaj göndermez.
Sürekli açık sunucu yoksa `CRYPTO_BOT_ROLE` repository variable'ını `primary`
yaparak bu iş akışını tek gönderici hâline getirebilirsiniz — o zaman tempo
saatlik olur ve yalnız `1h` modelinin bildirim penceresi tutar.

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
- Tazelik kapısı hedef mumda kalan süreye bakar; geciken bir koşu `5m` sinyalini
  göndermek yerine düşürür. `5m` için pencere 2 dakika, `15m` için 6, `1h` için
  24 dakikadır. Sürekli açık sunucuda 60 saniyelik tarama hepsinin içine düşer;
  yalnız GitHub Actions ile çalışırken yalnız `1h` penceresi tutar.
- Çok sayıda model/parametre denemesi yapılmaz; yine de altı model birlikte incelendiği
  için tek bir iyi sonuca aşırı anlam yüklenmemelidir.
- `%50` yakınındaki kısa vadeli piyasa yönü normaldir. Araştırma kapısını hiçbir model
  geçmezse doğru davranış işlem sinyali göndermemektir.
- Bu yazılım yatırım tavsiyesi değildir ve otomatik alım-satım yapmaz.

Veri istemcisi Binance'in resmî salt-okunur market-data kökünü, bildirim istemcisi
Telegram Bot API `sendMessage` metodunu kullanır.

Resmî teknik dayanaklar: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
ve [Telegram Bot API sendMessage](https://core.telegram.org/bots/api#sendmessage).
