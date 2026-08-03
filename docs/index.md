---
title: "Trade Lab — BTC/ETH olasılık araştırma botu"
---

# Trade Lab

BTC ve ETH için, kapanmış Binance mumlarından kalibre olasılık üreten bir
araştırma botu. Emir vermez, borsa hesabına bağlanmaz, API anahtarı kullanmaz.

Bu sayfa neyi kurduğumuzu, **neden öyle kurduğumuzu** ve ölçtüğümüzde ne
çıktığını anlatır. Sonuçlar olumlu değil; onları da olduğu gibi yazdık, çünkü
bir araştırma kaydının değeri doğruladıklarında değil, elediklerindedir.

---

## Bir cümlelik özet

Altı ayrı hipotez test edildi, **altısı da reddedildi**; hiçbir model
maliyet sonrası pozitif beklenti üretmiyor, ve bot bu yüzden hiçbir işlem
sinyali göndermiyor — bu bir arıza değil, sistemin doğru çalışmasıdır.

---

## 1. Sistem nasıl çalışıyor

```
Binance kapalı mumlar
   → belirteçler (9 gösterge, yalnız o an bilinen değerlerden)
   → hedef: üçlü bariyer (önce hangi tarafa %1 gidilir, 24 saat içinde)
   → kronolojik walk-forward + Platt kalibrasyonu
   → maliyet sonrası beklenti kapısı
   → Telegram (iki seviye) + canlı karne
```

Varsayılan olarak altı bağımsız model var: `BTCUSDT` ve `ETHUSDT` × `5m`,
`15m`, `1h`. Her biri kendi ayrı düzenlileştirilmiş lojistik regresyonu. İzlenen
semboller `CRYPTO_SYMBOLS` ile değiştirilebilir; Bonferroni düzeltmesi model
sayısından türetildiği için evren büyüdükçe kapı kendiliğinden sıkılaşır.

**Veri:** Binance'in salt-okunur kline uç noktasından yalnızca **kapanmış**
mumlar; varsayılan olarak **sürekli vadeli sözleşme**, çünkü işlem yapılan
enstrüman o. Spot fiyat yakın seyreder ama farklı bir enstrümandır, bu yüzden
önbellek dosya adı piyasayı içerir — biri diğeri sanılamaz. Açık mum asla
kullanılmaz. Önbellekte OHLCV'ye ek olarak `quote_volume`,
`trade_count` ve `taker_buy_base` da saklanır.

**Belirteçler (9):** ATR'ye bölünmüş 1/3/12 mum momentumu, EMA(8/21) farkı,
RSI(14), Bollinger(20) z-konumu, log ATR(14) yüzdesi, 20 mumluk log-hacim
z-skoru, mum içi alıcı/satıcı baskısı.

Haber, sosyal medya, zincir verisi veya gelecekte oluşan hiçbir değer
kullanılmaz.

---

## 2. Hedef: neden "üçlü bariyer"

İlk kurulumda hedef **"bir sonraki mumun kapanışı daha yüksek mi"** idi. Bu
hedefin sessiz bir kusuru var: bir baz puanlık sürüklenmeyi gerçek bir hareketle
aynı olay sayar. Böylece bir model isabetli görünürken komisyona para
kaptırabilir — nitekim kaptırıyordu.

Yeni hedef, o mumda açılan bir işlemin **önce hangi bariyere değdiğidir**:

- **Kâr al / zarar kes:** girişten `±100` baz puan (hedeflenen `%1`'lik işlem)
- **Süre bariyeri:** en fazla 24 saat; o zamana kadar değilmezse piyasadan kapanır
- **Aynı mumda iki seviye birden görülürse** hangisinin önce geldiği mum
  verisinden bilinemez → **her iki yön için de zarar** sayılır

Son madde önemli: bir tacir, kanıtlayamadığı iyi sonucu varsayamaz.

### Bariyer genişliği tek başına işi belirler

Simetrik bariyerde başabaş isabet oranı:

> **gerekli isabet = %50 + maliyet / (2 × bariyer)**

| Bariyer | Maliyet | Gereken isabet |
|---|---|---|
| 40 bps | 20 bps (spot taker) | **%75** — pratikte imkânsız |
| 100 bps | 10 bps (vadeli taker) | **%55** — ulaşılabilir bir bar |

İlk kurduğumuz bariyer (maliyetin 2 katı) hiçbir modelin geçemeyeceği bir bar
koyuyordu. Bunu proje sahibi fark etti: "%1'lik işlem yapacağım, vadelide
komisyon çok daha küçük kalmaz mı?" Haklıydı — ve düzeltme barı `%75`'ten
`%55`'e indirdi, örneklem 20 kat büyüdü.

---

## 3. Doğrulama protokolü

Veri **asla karıştırılmaz**. Her fold şu sıradadır:

```
geçmiş train → embargo → daha yeni kalibrasyon → embargo → daha yeni test
```

Embargo **tam bir etiket ufku** kadardır. Bariyer etiketi ileriye 288 mum (5m
için 24 saat) baktığı için tek mumluk embargo, eğitim setinin komşusunun
puanlandığı fiyat hareketini görmesine izin verirdi.

### Bildirim kapısı

Bir modelin işlem sinyali gönderebilmesi için **hepsi** gerekir:

1. En az 100 yüksek güven OOS örneği
2. Yüksek güven yön doğruluğu ≥ `%53`
3. Tüm modeller birlikte değerlendirildiği için Bonferroni aile-düzeltmeli
   `%95 Wilson` alt sınırı `%50` üzerinde — düzeltme model sayısından türetilir,
   böylece evren büyüdükçe bar kendiliğinden yükselir
4. Brier skoru, yalnız tarihsel yukarı oranını kullanan tabandan iyi
5. ECE ≤ `%10`
6. **Maliyet düşüldükten sonra sinyal başına beklenti pozitif**
7. **Bu beklentinin blok bootstrap `%95` alt sınırı sıfırın üstünde**

Son iki madde belirleyicidir. Yön isabetinin `%50` üzerinde olması tek başına
hiçbir şey ifade etmez: kazanan ve kaybeden işlemlerin büyüklükleri farklıdır.

### Blok uzunluğu neden etiket ufkundan büyük olmalı

Yüksek güven sinyalleri aynı seansta kümelenir. Bağımsız örnek varsayan güven
aralığı bu yüzden fazla dardır. Ama **takvim günü de yetmez**: `5m` etiketi 288
mum ileri baktığı için sonuç penceresi tam bir gün kaplar, komşu günler örtüşür.

Blok artık veriden türetilir: **en az etiket penceresinin iki katı, asla bir
günden kısa değil.** Bu hatayı bir deney sırasında bulduk — ayrıntısı aşağıda.

---

## 4. Bugünkü sonuçlar

Bariyer `±100` bps, maliyet `10` bps (Binance USD-M vadeli taker), gereken
isabet `%55`. Ölçüm 365 günlük veri, tamamen OOS dilimlerden:

| Model | OOS mum | Çözülen | İsabet (n) | Brüt bps | Net bps | Net %95 aralık | Kapı |
|---|---:|---:|---:|---:|---:|---:|:---:|
| BTCUSDT 5m | 41.349 | %84 | **%53.89** (3.333) | +5.08 | −4.92 | −28.82 / +17.43 | KALDI |
| BTCUSDT 15m | 13.778 | %89 | %48.87 (839) | −2.13 | −12.13 | −41.55 / +14.74 | KALDI |
| BTCUSDT 1h | 3.439 | %97 | %46.51 (86) | −8.02 | −18.02 | −63.57 / +32.86 | KALDI |
| ETHUSDT 5m | 41.349 | %98 | %50.08 (4.617) | −0.02 | −10.02 | −29.60 / +6.86 | KALDI |
| ETHUSDT 15m | 13.778 | %98 | %50.41 (1.817) | +1.04 | −8.96 | −27.38 / +6.82 | KALDI |
| ETHUSDT 1h | 3.439 | %100 | %47.50 (400) | −8.72 | −18.72 | −43.23 / +2.41 | KALDI |

**Hiçbiri geçmiyor.** En iyisi `BTCUSDT 5m`: %53.89 isabet, gereken %55. Aradaki
1,1 puan küçük görünüyor ama altı farklı yoldan kapatmaya çalıştık ve hiçbiri
tutmadı.

Bir de rahatsız edici bir referans var: bariyer hedefinde yukarı tarafın önce
görülme oranı `%52.5`. Yani hiç düşünmeden "yukarı" diyen bir kural, modele
yakın isabet veriyor.

---

## 5. Denenen ve elenen hipotezler

Hepsi aynı walk-forward dilimlerinde, aynı kapıyla ölçüldü. **Her fikir
önceden yazıldı, bir kez ölçüldü, sonucu kabul edildi.**

### 5.1 Emir akışı — reddedildi

Binance her mumla birlikte taker alım hacmini gönderir. Bir mum pasif alımla da
yeşil kapanabilir, agresif satışa denk gelen bir alışla da; OHLCV bu ikisini
ayırt edemez, taker hacmi edebilir. Umut verici görünüyordu.

Dört form denendi: ham denge, 12 mumluk ortalama, 20 mumluk z-skoru, 3 mumluk
değişim, artı ortalama işlem büyüklüğü. **18 model karşılaştırmasının 16'sında
net beklenti kötüleşti**, en kötüsü −16,4 bps.

*Neden:* agresör ayrımı mumun kendi hareketiyle eşzamanlı, yani büyük ölçüde
`candle_pressure`'ın tekrarı. Bilgi katmadan varyans ekliyor.

### 5.2 Gradient boosting — reddedildi, ama bir hata buldurdu

`HistGradientBoosting` ilk bakışta atılım gibi görünmüştü:

| Model | İsabet | Net bps | Ayrı hafta | Blok alt sınırı |
|---|---|---|---|---|
| ETHUSDT 5m | **%70.6** | **+27.3** | 10 | **−16.0** |
| ETHUSDT 15m | %60.2 | +8.4 | 10 | −26.2 |
| BTCUSDT 15m | %34.9 | −38.9 | 9 | −55.9 |

1.594 sinyal görünüyor ama yalnız **10 ayrı haftaya** düşüyor. Kümelenmeye göre
düzeltince artı kayboluyor. Üstelik isabetin modeller arasında %34,9 ile %70,6
arasında salınması, sinyalin değil varyansın imzası.

**Bu deney kapıdaki gerçek bir hatayı ortaya çıkardı:** bootstrap gün
bloklarıyla yapılıyordu, oysa etiket penceresi bir günü kaplıyor. Gün blokları
bağımlılığı kırmıyordu. Düzeltildi — ve deney kaybetmesine rağmen altyapıyı
sağlamlaştırdı. `scikit-learn` bağımlılığı eklenmedi.

### 5.3 Funding oranı — reddedildi

Funding, uzun pozisyonların kısalara ödediği bedeldir; aşırı bir oran, kalabalık
tarafı ve sıkışma riskini gösterir — fiyat geçmişinde olmayan bir bilgi.

Üç form: son ödenen oran, 30 ödemelik z-skoru, son 24 saatin toplamı. Hizalama
`merge_asof(direction="backward")`, yani bir mum yalnız kapanışından **önce**
ödenmiş funding'i görür. **Altı modelin beşinde kötüleşti**; tek iyileşende bile
alt sınır −34,8 bps'de kaldı.

### 5.4 BTC-ETH öncülük — reddedildi

Her modele diğer sembolün 1 ve 3 mumluk momentumu ve ikisinin farkı eklendi,
`close_time_ms` üzerinden birebir hizalanarak (hizalama %100 tuttu, teknik
aksaklık yok). **Altı modelin beşinde kötüleşti.**

### 5.5 Oynaklık rejimi filtresi — önerme yanlış çıktı

Fikir: `%1`'in ulaşılamayacağı sakin rejimlerde sinyal üretmeyelim, ölü ağırlık
taşımayalım. Filtre: `ATR × √ufuk ≥ bariyer`.

Ölçüm önermeyi çürüttü: bu koşul zaten zamanın **%91-100**'ünde sağlanıyor.
`5m` için `ATR × √288 ≈ 204` bps, bariyerin iki katı. `%1` hedefi sakin
dönemlerde de ulaşılabilir. Filtrenin fiilen elediği tek yerde
(`BTCUSDT 5m`, sinyallerin %26'sı) sonuç −4,9'dan −9,7 bps'ye geriledi.

### 5.6 Altcoin evreni — reddedildi, ve önce beni yanılttı

Proje sahibinin bir aylık vadeli işlem geçmişi, botun **yanlış piyasaları**
izlediğini gösterdi: 18 sembolde işlem yapılmış, bot ikisine bakıyordu ve
`ETHUSDT` sonucun `%0.5`'iydi. Kârın `%83`'ü iki altcoin'den geliyordu.

İlk ölçüm umut verdi. Onun işlem yaptığı 8 altcoin'de dördü `%55` barının
üzerindeydi, ikisinin net beklentisi pozitifti — `BTC` ve `ETH` ise listenin en
altındaydı.

**Ama sembolleri, onun kârlı işlem yaptığı yerlerden seçmiştim.** Yani listeye
geriye dönük olarak "hareketli çıkmış" varlıklar koymuş oldum. Bu, tam olarak
kapının engellemeye çalıştığı hatanın bir biçimi.

Temiz test için evren **sonuçlara bakılmadan** sabitlendi: en az 400 gün önce
listelenmiş USDT sürekli vadeli sözleşmeler, güncel hacme göre ilk 40. Hepsi
ölçüldü, hiçbiri sonradan elenmedi.

| | Ölçülen | Kenar yoksa beklenen |
|---|---|---|
| Kapıyı geçen | **0 / 40** | 0 |
| Medyan isabet (n≥100) | `%51.35` | `%50` |
| Medyan net | `-10.37` bps | `-10.00` bps |
| Örneklem-ağırlıklı net | `-8.20` bps | `-10.00` bps |
| Alt sınırı sıfırı geçen | `0 / 14` | 0 |

Ağırlıklı net ile "kenar yok" varsayımı arasındaki fark `1.8` bps — komisyonu
bile karşılamayan bir gürültü farkı. `%55` barını geçen sembol oranı, yanlı
seçimde `4/8` iken önceden belirlenmiş evrende `4/14`'e düştü ve hiçbiri güven
aralığı sınavını geçemedi.

Küçük örneklem tuzağı da görünür oldu: `n<100` olan 13 sembolün 5'i pozitif net
gösteriyor, medyan örneklemleri 36. `LINKUSDT` `%68` isabet (n=19), `NEARUSDT`
`%100` (n=3). Kapının 100 eşiği tam olarak bunları elemek için var.

Bu deney iki düzeltme getirdi: izlenen semboller artık ayarlanabilir, ve
Bonferroni düzeltmesi model sayısından türüyor — sabit `z` 40 sembolde düzeltmeyi
sessizce zayıflatırdı, ki bu olmayan kenarları var gösteren yöndür.

### 5.7 Açık pozisyon — test **edilemedi**

Binance `openInterestHist` uç noktası yalnız son ~30 günü tutuyor; 60 gün öncesi
`HTTP 400` veriyor. Bir yıllık walk-forward için geçmiş yok.

Bu yüzden bot **ileriye doğru kayıt tutuyor**: her turda veriyi çekip
`data/{SEMBOL}_open_interest.csv` dosyasına ekliyor. Bu kayıt hiçbir modeli
beslemiyor, hiçbir iddia taşımıyor — amacı, birkaç ay sonra sorunun
sorulabilmesi. Fiyattan türemeyen, henüz denenmemiş tek bilgi kaynağı.

---

## 6. Bildirim tasarımı

Altı modelin hiçbiri kapıyı geçmediği için işlem sinyali gelmiyor. Ama kanalın
sessiz kalması "bot bozuk" ile "geçen model yok" arasındaki farkı gizlerdi. Bu
yüzden iki seviye var:

| Seviye | Ne demek | Ne zaman |
|---|---|---|
| **İŞLEM ADAYI** | Maliyet sonrası beklentisi pozitif ve alt sınırı sıfırın üstünde | Kapıyı geçen modellerden, hedef mumun en az %60'ı önündeyken |
| **GÖZLEM** | Model tahmin üretti ama işlem beklentisi kanıtlanmadı | Altı modelin tamamı için **günde bir** özet |

Ayrıca bot sorulduğunda cevap verir: `/durum` (altı modelin anlık durumu),
`/performans` (canlı karne), `/kisiler`, sahip için `/ekle` ve `/sil`.

**Tazelik kapısı:** bir tahmin ancak hedef mumun en az %60'ı önündeyken
gönderilir. Kapanmış bir mumun kapanışını tahmin etmek tahmin değildir.

---

## 7. Canlı karne

Backtest modeli ölçer; canlı karne **botun kendisini** ölçer. Gönderilen her
sinyal bariyeri çözülene kadar bekletilir — kâr al, zarar kes ya da süre dolması
— ve backtest'in puanladığı şekilde puanlanıp kalıcı bir deftere yazılır.

Sonraki mumun kapanışına bakmak, modele sorulandan başka bir soruyu ölçerdi.
**Yanlış şeyi ölçen bir ileri test, hiç test olmamasından kötüdür: kanıt gibi
görünür.**

### İstatistiksel güç — dürüst beklenti

Bariyer ufku 24 saat olduğu için model başına günde en fazla bir bağımsız
gözlem çıkar; semboller ve zaman dilimleri birbiriyle ilişkili olduğundan
gerçekte günde ~2-3 bağımsız gözlem.

| Süre | Ne öğreniriz |
|---|---|
| 1 ay | Yalnız kaba arıza tespiti |
| 3 ay | Yön eğiliminin işareti |
| 6+ ay | %5 puanlık kenarı ayırt edebilecek güç |

Yani canlı karne tek başına modeli **doğrulayamaz**. İşi, backtest'in yalan
söyleyip söylemediğini yakalamaktır — geriye dönük optimizasyona hiç maruz
kalmamış tek ölçüm olduğu için değerlidir.

---

## 8. Altyapı ve neden öyle

**Nerede çalışıyor:** Oracle Cloud Always Free sanal makinesinde `systemd`
servisi olarak, 7/24, 60 saniyelik tarama.

**Neden GitHub Actions değil:** ölçüldü — GitHub, istenen `*/5` zamanlamanın üç
saatte ~35 koşusundan yalnız **2**'sini çalıştırdı; fiili tempo saatte bir. Buna
ek olarak private depoda her iş tam dakikaya yuvarlanarak 2.000 dakikalık aylık
kotadan düşüyor; dakikada bir koşu ayda ~43.000 dakika eder ve 7/24 döngü
Actions kullanım şartlarına aykırıdır. Actions artık **yedek (standby)** rolde:
araştırmayı ve paneli güncel tutar, mesaj göndermez.

**Tek gönderici kuralı:** aynı anda iki `primary` kopya çalışırsa her uyarı iki
kez gider (teslimat makbuzları ayrı disklerde) ve Telegram eş zamanlı
`getUpdates` çağrılarını `409` ile reddettiği için komutlar kaybolur. Rol
bayrağı bunu yapısal olarak imkânsız kılar.

**En-fazla-bir-kez teslimat:** sunucu cevap verip reddettiyse (403, 400) mesaj
kesinlikle gitmemiştir → tekrar denenir. Zaman aşımı olduysa gitmiş olabilir →
tekrar denenmez. Mükerrer uyarı yollamaktansa bir raporu kaçırmak yeğdir.

**Verimlilik:** bir modelin girdisi ancak yeni mum kapandığında değişir ve o an
tam olarak bilinir, dolayısıyla tekrar kullanım bir yaklaşıklık değil birebir
aynı sonuçtur. Altı model için tur süresi **2,77 sn → 0,0002 sn**; saatteki ağır
hesap 360'tan 34'e indi.

---

## 9. Yol boyunca bulunan hatalar

Bu bölüm sonuçlar kadar önemli, çünkü hepsi sessizce yanlış sonuç üretiyordu.

| Hata | Etkisi |
|---|---|
| Kapı yön isabetine bakıyordu, kâra değil | %63 isabetli model komisyondan sonra zarardaydı; kapıyı geçen iki modelin ikisi de negatif beklentiliydi |
| Actions cache anahtarı saatlikti | `actions/cache` birebir anahtar isabetinde kaydetmeyi atladığı için 15m sinyali üç kez gidiyordu |
| Tazelik kuralı 3 mum izin veriyordu | Geciken koşu, **zaten kapanmış** mumun kapanışını tahmin ediyordu |
| Bariyer etiketinde embargo tek mumdu | Eğitim seti, komşusunun puanlandığı fiyat hareketini görüyordu |
| Bootstrap gün bloklu idi | 24 saatlik etiket penceresinde günler örtüşüyor; anlamlılık olduğundan büyük görünüyordu |
| 403 kalıcı sessizliğe yol açıyordu | Kesin ret "sonuç bilinmiyor" sayılıyor, yetki düzeltilse bile bir daha denenmiyordu |
| `update.sh` yeni testleri eski kütüphaneye karşı koşuyordu | Kütüphaneyi değiştiren hiçbir sürüm bu kapıdan geçemezdi |
| Komut yanıtı sessizce düşüyordu | Bozuk `/durum` ile hiç gönderilmemiş `/durum` ayırt edilemiyordu |
| Canlı karne yanlış hedefi ölçüyordu | Bariyer modelinin sonuçları sonraki mum kapanışıyla puanlanıyordu |
| Saatler UTC yazılıyordu | 14:19 UTC damgası, 17:19 gösteren telefonda üç saat eski görünüyordu |

---

## 10. Sınırlar

- Veri kaynağı tek borsa (Binance); başka borsada fiyat ve likidite farklı olabilir.
- Backtest komisyonu hesaba katar ama **kaymayı ve emir gerçekleşmesini ölçmez** —
  gerçek maliyet raporlanandan yüksektir.
- Altı hipotez denendi; her ek deneme, şans eseri barı geçen bir şey bulma
  ihtimalini artırır. Bonferroni düzeltmesi altı modeli kapsar, **kaç fikir
  denediğimizi kapsamaz**.
- `%50` yakınındaki kısa vadeli piyasa yönü normaldir. Kapıyı hiçbir model
  geçmezse doğru davranış sinyal göndermemektir.
- Bu yazılım yatırım tavsiyesi değildir ve otomatik alım-satım yapmaz.

---

## 11. Sırada ne var

Kendiliğinden ilerleyen iki şey var:

1. **Açık pozisyon verisi** birikiyor. 3 ayda ilk bakış, 6 ayda gerçek test.
   Fiyattan türemeyen, henüz denenmemiş tek kaynak.
2. **Canlı karne** birikiyor ve artık doğru soruyu ölçüyor.

Kalan ucuz fikirler tükendi. Daha fazla varyant denemek, er ya da geç şans eseri
geçen bir şey bulmak demektir — ve o noktada gerçek mi tesadüf mü ayırt etmek
mümkün olmaz. **Altı temiz ret, arayıp bulunacak bir "başarı"dan daha güvenilir
bir bilgidir.**

Sonuncusu bunun neden böyle olduğunu da gösterdi: yanlı seçilmiş sekiz sembol
"altcoin'lerde kenar var" dedi, önceden belirlenmiş kırk sembol demedi. Aradaki
tek fark, listeyi kimin ve neye bakarak yaptığıydı.

---

## Çalışma ilkesi

Bu projede tek bir kural her şeyin önünde geldi:

> Ölçülmemiş bir iddia yayınlanmaz; ölçülen sonuç, hoşa gitmese de yayınlanır.

Başlangıçta %63 isabetli görünen bir model vardı ve kimse komisyondan sonra para
kaybettirdiğini söylemiyordu. Bugün altı modelin de neden sinyal üretmediğini,
rakamıyla ve gerekçesiyle biliyoruz. Aradaki fark, bu projenin asıl çıktısı.
