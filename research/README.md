# Trade3 hedef dokunuşu araştırması

`long_discovery_study.py` yeni LONG keşif kurallarını ayrı saatlik önbellekte
keşif amaçlı tekrar oynatır. Çalıştırma:

```powershell
.\.venv\Scripts\python.exe research/long_discovery_study.py state/scalp/experiments/long-discovery-v1
```

Önce `python run.py long-scout` ile yerel fiyat önbelleği ve evren anlık
kopyası oluşturulmalıdır. Bugünkü evren seçimi nedeniyle bu pilotta hayatta
kalanlar yanlılığı vardır; gelecekteki başarı olasılığı veya bağımsız test
diye kullanılmaz. Canlı sessiz deney bundan sonra yeni 5m kayıtları toplar.

`target_touch_study.py` çevrimdışı bir araştırma aracıdır. Canlı bot ayarlarını
değiştirmez, Telegram mesajı göndermez ve model terfisi/deploy yapmaz.
Trade1 verilerine veya koduna yazmaz.

## Yeniden çalıştırma

Projenin bağımlılıkları kurulu Python ortamında, depo kökünden:

```powershell
.\.venv\Scripts\python.exe research/target_touch_study.py artifacts/research/target-touch-20260928 --cutoff 2026-09-29T17:55:00+00:00
```

Çalışma klasöründe Trade3 `ledger.jsonl` anlık kopyası ve `prices/*.json.gz`
Binance USD-M 5 dakika fiyat önbelleği gerekir. `--fetch` yalnız bu yerel
araştırma önbelleğine kamuya açık fiyatları indirir. Ham veriler ve raporlar
`artifacts/research/` altında Git tarafından dışlanır; otomatik yayımlanmaz.

Çıktılar: `report.md`, `study.json`, `labeled-setups.json`.
JSON; eğitim/test bloklarını, katsayıları, satır tahminlerini ve veri/kod
SHA256 değerlerini içerir. Kodun sabit kohortu 27–28 Eylül 2026'daki 17
kurulumdur; farklı bir çalışma için tarih aralığını bilinçli seçmek gerekir.

## Değerlendirme sınırları

- Bir coin/mum kurulumu tek örnektir. Aile ve 15/30/60 dakika kayıtları
  örnek sayısını yapay olarak artırmaz.
- Referans fiyat sinyal mumunun kapanışıdır; defterdeki sonraki mum açılışı
  tahmin girdisi değildir. Sonraki 24 saatin hedef dokunuşları ölçülür.
- Eğitim, kalibrasyon ve test zaman sıralıdır; sonuç pencereleri ayrılır.
  Eksik/henüz tamamlanmamış fiyat verisi başarısız sonuç diye etiketlenmez.
- LONG/SHORT tarafı yalnız model olasılığından seçilir; sonradan kazanan
  yöne dönülmez. %2/%3/%5 dokunuşları gerçek işlem kârı anlamına gelmez.
- 29 Eylül raporunda birincil filtre geniş testte %2 isabetini
  **%58,7'den %54,9'a düşürdü**. Bu sürüm canlı kullanım için doğrulanmadı.
  17 örnekteki 9/12 isabet tek başına aktivasyon gerekçesi değildir.
- Bu sonuçlara bakarak geliştirilecek sonraki model, yeni ve bağımsız
  tarihlerde doğrulanmalıdır. Buradaki kohort artık görülmemiş test sayılmaz.
