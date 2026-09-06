# Trade3 teknik ve ölçüm incelemesi — 7 Eylül 2026

## Sonuç

Trade3'ün dashboard arızası giderildi. Son değişikliklerde veri doğruluğu,
Telegram teslimatı ve sonuç kaydıyla ilgili ek hatalar bulundu ve bu sürümde
regresyon testleriyle düzeltildi. Trade1 koduna, ayarlarına ve sunucusuna dokunulmadı.

Bot bir araştırma/bildirim sistemi; emir gerçekleştiren bir işlem motoru değil.
Çalışır olması, yüksek skor veya yüksek görünen bir yüzde, kârlılığın doğrulandığı
anlamına gelmez. Hedefimiz hatası ölçülebilen, güvenilir ve kontrollü geliştirilen
bir bot olmalı; kusursuzluk ya da kazanç garantisi verilemez.

İnceleme kapsamı: dashboard ve yayın akışı, piyasa verisi, evren manifesti,
scalp stratejileri/istatistikleri, BTC/ETH özellik-model-backtest hattı, sonuç
takibi, Telegram erişim/teslimat, servis döngüsü, dağıtım ve testler.
Bu bir bağımsız sızma testi veya stratejinin yeniden yapılmış 365 günlük
backtesti değildir. Geçmiş sonuçlar yeniden yazılmadı; değişen hesapların
geçmişe dönük etkisi ayrı bir yeniden değerlendirme çalışması gerektirir.

Yerel doğrulama: 169 Python testi ve dashboardun Node regresyon testi geçti.
Yeni testler kısmi teslimatı, üyelik değişiminde tekrarları, NaN/Infinity verisini,
eksik mumları, mükerrer örnekleri, disk yazma hatasını ve sonuç kaydı tekrarını
kapsıyor. Canlı tarayıcıda onarılmış sayfanın araması ve dokunuşları doğrulandı.

## Canlı kanıtlar

Salt okunur sunucu ölçümü: 6 Eylül 2026 22:06 UTC / 7 Eylül 01:06 Türkiye saati.
O anda sunucudaki sürüm `c3bb0c9`; dashboard onarımı `22470d1` Pages üzerinde yayında.

- Servis aktif; son taramalar 89/89 taze piyasa, BULL rejimi gösteriyor.
- Scalp gözlem defterinde 8.694 satır, coin/zaman/ufuk düzeyinde 7.392 farklı
  gözlem var; veri 8 farklı UTC gününü kapsıyor. Farklı ufuklar da bağımsız
  işlem sayılmaz. Aile satırlarının fazla olması tek başına veri bozukluğu değil;
  bunları tek bir kurulum için bağımsız örnek gibi toplamak hataydı.
- Dinamik hedef/stop: 6 sonuç = 1 TARGET, 5 STOP; ortalama net −46,07 bps
  (yaklaşık −%0,46). Bunlar simülasyon sonuçlarıdır; gerçek hesap getirisi değil.
- Bildirimli sabit hedef kayıtları: %2'de 5 dokunuş; %3'te 3 dokunuş ve 1
  süresi dolmuş kayıt; %5'te 2 süresi dolmuş kayıt. Açık kayıtlar bu toplamların
  dışında. Bu farklı ve henüz tamamlanmamış grupları toplayıp başarı oranı
  çıkarmak veya %5'i eski %2 sonuçlarıyla kıyaslamak yanıltıcı olur.
- Canlı dashboardda STRK örneği hem %2/%3 dokunuşu hem daha erken STOP gösteriyor.
  Çelişki yok: ayrı soruları ölçüyorlar. Dokunuş, stoplu bir işlemin kazandığını
  kanıtlamıyor.
- Üye kayıt dosyasının sunucu izni `0600`. Üyelik yönetimi sahip kimliğiyle,
  komutlar özel sohbet ve yetki kontrolleriyle korunuyor.

## Bu incelemede düzeltilenler

| Sorun | Düzeltme ve doğrulama |
|---|---|
| Dashboard JavaScript sözdizimi hatası ve HTML sonuna karışmış script parçaları | Tek HTML belgesi ve ayrı `docs/scalp.js`; gerçek tarayıcıda tablo, arama ve dokunuşlar doğrulandı. |
| Python testleri kırık arayüzü fark etmiyordu | Bağımlılıksız Node testi: sözdizimi, satırlar, filtreler, eksik/boş/hatalı veri ve zararlı metin. CI ve Pages yayınından önce çalışıyor. |
| Dashboardda belirsiz “Bildirim filtresi %...” kartı | Tek yüzde yerine sonuç/kademe sayıları; dokunuş, stoplu sonuç, ilk sinyal teslimatı ve tahmin yüzdesi arasındaki fark açıklanıyor. Kartlar tüm veriyi, filtreler tabloyu kapsar. |
| Telegram alıcıların yalnız bir kısmına gönderse de SENT dönüyordu | `PARTIAL` ayrı sonuç. Başarılı alıcıya tekrar gönderilmez; reddedilen alıcının hedef teslimatı tamamlanmış sayılmaz. Kısmen teslim edilmiş ilk sinyalin sonuç takibi korunur. |
| Yeni üye katıldığında eski, tekrar kontrol edilen mesajlar yeniden gönderilebiliyordu | Mesaj bazında ilk alıcı listesi kaydedilir; sonraki üyeler geçmiş mesajın tekrarına eklenmez. Erişimi kaldırılmış alıcılar yeniden denemeden çıkarılır. |
| Eksik mumlar dinamik hedefte yanlış giriş/ilk hedef veya erken süre sonu üretebiliyordu | Girişten itibaren kesintisiz veri gerekir. Boşluktan önce kanıtlanmış ilk dokunuş geçerli; boşluk üzerinden hedef/stop/süre sonucu üretilmez. Sabit zamanlı değerlendirmede ufkun kayması da engellendi. |
| Aynı hareketin B1/F3 vb. ailelerde bulunması kurulum örnek sayısını şişiriyordu | Coin/zaman/ufuk tekilleştirmesi; farklı maliyet varsa ihtiyatlı net değer. Yerel örnek seçimi artık ilgili ufkun satırları üzerinden yapılıyor. |
| NaN/Infinity içeren sayısal piyasa verisi bazı kontrollerden geçebiliyordu | Sonlu sayı doğrulaması ve bozuk değer testleri. |
| Bekleyen kayıt sonuç defterine yazılmadan silinebiliyordu | Önce kalıcı sonuç yazımı, sonra bekleyen kaydın onayı/silinmesi; tekrar işlemeye dayanıklı anahtarlar, disk senkronizasyonu. Disk yazma hatası ve yeniden başlatma tekrarları test edildi. |

Yeni alıcı listesi her mesaj için tutulur; bir sinyal ile daha sonra oluşan hedef
mesajının alıcı listesini birbirine bağlayan tam bir teslimat kuyruğu henüz yok.
Telegram ağ zaman aşımında teslimat bilinemez; otomatik tekrar yerine UNCERTAIN
kalır. Bu sınırlamalar gizlenmemeli. JSONL iyileştirmesi tek yazıcılı servise
yöneliktir; çoklu süreçler için veritabanı işleminin yerini tutmaz.

## Öncelikli geliştirme planı

### 1. Ölçüm ve olasılık doğruluğu — en yüksek öncelik

Kaynak: `scalping.py: scalp_setup_assessment`, `scalp_setup_direction`,
`filter_scalp_notification_report`; `dashboard.py: build_dashboard_payload`.

Scalp yüzdeleri ailelerin geçmiş ileri-test getirilerinden türetiliyor; tam olarak
bu kurulumun veya %2/%3/%5 hedefinin olasılığı değil. Aileleri tekilleştirmek,
coinler/zamanlar arasındaki bağımlılığı veya tüm seçim yanlılığını çözmez.
Yeterli örnek yokken ham skora dönen bildirim yolu ve farklı rejimlere açılan
örnek havuzu sürüyor. Güven etiketi tek başına bir yayın engeli değil.

Yapılacaklar:

- Tekil kurulum kimliği, strateji/ayar/kod sürümü, sinyal anındaki değerlendirme
  havuzu, tespit ve teslimat zamanlarını değişmez olarak saklamak.
- “Belirli ufukta maliyet sonrası pozitif getiri”, “stop öncesi hedef” ve
  “24 saat içinde %2/%3/%5 dokunuş” için ayrı tahminler ve ayrı başarı tabloları.
- Tamamlanmış başlangıç kohortları, örnek sayısı, bağımsız gün/blok ve belirsizlik
  aralıklarıyla raporlamak. Bildirimli/sessiz, long/short ve rejimleri ayırmak.
- Scalp tarafına kronolojik eğitim/kalibrasyon/test ayrımı, ufuk kadar aralık,
  olasılık kalibrasyon eğrileri ve taban oran karşılaştırması eklemek.
- Yetersiz kanıtta “deneysel” yolu ile doğrulanmış sinyal yolunu ayırmak;
  varsayılan yayın politikasını kullanıcıyla açıkça kararlaştırmak.

Kabul ölçütü: aynı kurulum aile eklenince bağımsız örnek gibi çoğalmaz; her yüzde
hangi olay ve ufuk için hesaplandığını belirtir; test verisiyle ayar seçilmez.

### 2. Gerçekçi ileri test ve strateji karşılaştırması

Kaynak: `scalping.py: _first_touch_bracket`, `_dynamic_bracket_bps`;
`features.py`, `model.py`.

Sonraki mum açılışı varsayımsal giriş; gerçek mesaj o açılıştan sonra ulaşabilir.
Beş dakikalık mum içi sıra bilinmiyor; stop önceliği ihtiyatlı fakat tam çözüm
değil. MFE/MAE çıkış mumunun tamamını içeriyor, çıkış sonrası uçları da kapsayabilir.
Komisyon/kayma tahmini ve varsa fonlama gerçekleşen işlem maliyeti değildir.

Yapılacaklar: teslimat sonrası erişilebilir fiyatla ayrı simülasyon, dokunuşları
1 dakikalık veriyle ayrıntılandırma, boşluk/gap kayması ve fonlama senaryoları,
eşzamanlı ve aynı coinde çakışan sinyallerin ayrı işaretlenmesi. Mevcut strateji
ve aday sürüm aynı dönemde gölge testte karşılaştırılmalı. Hedef/stop mesafeleri
6 sonuca bakıp değiştirilmemeli.

Kabul ölçütü: yalnız isabet değil ortalama net getiri, kayıp büyüklüğü,
azami düşüş ve maliyet hassasiyeti birlikte raporlanır. Üstünlük kanıtlanmadan
aday sürüm otomatik olarak ana sürüm yapılmaz.

### 3. Sağlam teslimat ve kalıcı kayıt

Kaynak: `telegram.py`, `service.py`, `scalping.py`, `outcomes.py`.

SQLite/WAL tabanlı tekil sinyal, hedef ve alıcı teslimat tabloları; işlem içinde
kuyruk/onay güncellemesi; reddedilen, belirsiz, kısmi ve tamamlanan teslimatların
yönetici raporu. Hedef bildiriminin alıcıları ilk sinyali gerçekten alanlarla
ilişkilendirilmeli. Belirsiz teslimata otomatik kör tekrar yapılmamalı.
Geçersiz/eksik veri kayıtları sessizce atılmak yerine karantinaya alınmalı;
dashboardda VERİ EKSİK durumu ve nedeni görünmeli.

Kabul ölçütü: yeniden başlatma/kısmi disk hatası/403/zaman aşımı testlerinde
ölçüm kaybolmaz; tüm alıcılar için teslimat durumu açıklanabilir.

### 4. Servis sağlığı ve veri toplama

Kaynak: `service.py: serve`, `data.py: update_market_cache`, `openinterest.py`,
`universe.py`.

Tarama, model yenileme ve komutlar aynı döngüde. Borsa yavaşlığı/eğitim, menü
yanıtlarını ve bildirimleri geciktirebilir. Mevcut cache boşluk sonrası eski
mumları atıp kesintisiz kuyruğu tutuyor; geriye dönük veri onarımı eksik.

Yapılacaklar: sınırlı eşzamanlı veri toplama ve istek bütçesi, komut/teslimat
kuyruğunu ağır işlerden ayırma, son başarılı tarama ve teslimat gecikmesini
ölçme, takılı süreç ve eski veri uyarısı. Salt-okunur evren manifesti sürümlü
kalmalı; yeni/delist edilmiş piyasa durumu ve likidite kontrolü Trade3 içinde
yapılmalı. OI kaydı şu anda BTC/ETH odaklı; geniş evrene veri eklemek ancak
ölçülebilir bir hipotez için ve istek bütçesiyle yapılmalı.

Kabul ölçütü: komut yanıtı ve taze sinyal gecikmesi için ölçülen servis hedefleri;
veri kesildiğinde eski fiyatla yeni sinyal yok; kayıp mumların onarım izi var.

### 5. Güvenli sürüm ve kurtarma

Kaynak: `deploy/update.sh`, `deploy/crypto-forecaster.service`, `pyproject.toml`,
`.github/workflows/tests.yml`, `.github/workflows/cloud-bot.yml`.

Olumlu: root olmayan kullanıcı, systemd sınırlandırmaları, gizli env dosyası,
özel üye dosyaları, Pages için alan izin listesi ve kısıtlı SSH erişimi var.
Eksikler: bağımlılık sürümleri tam kilitli değil; güncelleme çalışan kurulumun
dosyalarını/ortamını değiştiriyor; rollback bağımlılıkları birebir geri getirmiyor.
Kod yedeği, durum/veri yedeği değildir. Mevcut inceleme sunucu dışı yedek ve
geri yükleme tatbikatını doğrulamıyor.

Yapılacaklar: kilitli bağımlılıklar, ayrı sürüm dizini ve sanal ortam, testten
sonra atomik geçiş, ilk başarılı tarama sonrası sağlık doğrulaması ve geri dönüş.
Şifreli veri/durum yedeği ve düzenli geri yükleme testi. CI'da desteklenen Python
sürümleri, gerçek tarayıcı smoke testi, lint ve bağımlılık güvenlik taraması.
Normal SIGINT çıkışındaki 130 kodu gerçek çökmeden ayrı gösterilmeli.

### 6. Daha sade ürün ve doğru erişim sınırı

Telegram: yön, sinyal fiyatı, ufuk, tanımlı olasılık, hedef/stop ve kısa güven
notu; ayrıntılar düğmede. Stop ve süre sonu da performans bölümünde hedef kadar
kolay bulunmalı. Dashboard mobilde özet kart + açılır detay satırı, coin/tarih/
strateji/bildirim filtreleri, tamamlanmış dönem karşılaştırması sunmalı.

GitHub Pages sayfası herkese açık; Telegram üyeliği bu sayfaya erişimi sınırlamaz.
Kişisel üye verisi yayınlanmıyor, ancak sinyaller yayınlanıyor. Sinyaller de özel
kalacaksa ayrı kimlik doğrulamalı dashboard gerekir; bu ürün/erişim tercihi mevcut
incelemede varsayılarak değiştirilmedi.

## Önerilen çalışma sırası ve güncelleme aralığı

1. Bu onarım sürümünü doğrula; geçmiş kayıtları koru.
2. Önce ölçüm/teslimat defteri ve veri-eksik durumunu tamamla.
3. Strateji sürümlerini dondurup ayrı ileri test biriktir; günlük sağlık kontrolü,
   haftalık performans/kalibrasyon raporu üret.
4. Yeni modeli haftalık değerlendirmek uygundur; haftalık otomatik olarak
   değiştirmek zorunlu olmamalı. Veri ve rejim çeşitliliği yetersizse eskisini koru.
5. Yeterli bağımsız gözlem ve maliyet sonrası üstünlük sağlandığında kontrollü
   sürüm geçişi yap. İlk planlama penceresi 4–8 hafta olabilir; bu süre tek başına
   istatistiksel yeterlilik veya kâr kanıtı değildir.

Mevcut haftalık araştırma döngüsü BTC/ETH model yenilemesidir; tüm scalp
stratejilerini kendiliğinden optimize eden bir döngü değildir. Bu rapor yeni bir
zamanlayıcı oluşturmaz veya mevcut Trade1 işlerini değiştirmez.
