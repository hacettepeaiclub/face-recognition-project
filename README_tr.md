# Yüz Tanıma Uygulaması

## Gereklilikler

Python 3.11

## Yapı

```console
- celebrities_images  # Supported file types are: `jpg`, `jpeg`, `png`
    - Angelina Jolie  # Folder
        - 1.jpg
        - 2.jpg
        - 3.jpg
    - Brad Pitt  # Folder
        - 1.jpg
        - 2.jpg
        - 3.jpg
- app.py  # Contains the application
- models  # Contains the models that are required to run the application
```

## Yükleme

- Sanal ortam (virtualenv) oluştur.

- Gerekli olan tüm kütüphaneleri requirements.txt dosyasından yükleyin

- Gerekli kütüphaneleri yükledikten sonra [google drive](https://drive.google.com/drive/folders/1arXbBb_MA20MIgFr0Ap6euUjl-YGjqiD?usp=sharing) adresimizden modelleri indirin

- İstediğiniz resimleri projenin yapısına uygun bir şekilde 'celebrities_images' dizinine yerleştirin

- Email kısmına kendi email ve şifrenizi yazın. Email kısmında küçük bir farklılık var. Ordaki şifreyi kendi gmail şifrenizi kullanamıyorsunuz.

- App Password Kullan (Tavsiye Edilen): SMTP üzerinden e-posta göndermek için daha güvenli bir yol budur. Bu adımları izleyin:

E-posta hesabınıza giriş yapın. Hesap güvenlik ayarlarına gidin."Uygulama Parolaları" veya "Uygulama Parolası Oluştur" seçeneğini arayın. Uygulamanız için (bu durumda Python betiğiniz için) özel bir Uygulama Parolası oluşturun.Python kodunuzdaki 'your_password' ifadesini oluşturulan Uygulama Parolası ile değiştirin.

## İndirmeler

- Modelleri bu linkten indirin [google drive](https://drive.google.com/drive/folders/1arXbBb_MA20MIgFr0Ap6euUjl-YGjqiD?usp=sharing).

## Çalıştırma

- Dosya dizinine geldinten sonra terminali açın ve `python app.py` yazıp enter tuşuna basın.

## Yazarlar

- [Samet Emin Özen](mailto:sameteminozen2@gmail.com)
- [Barış Çelik](mailto:bariscelikww@gmail.com)

## Lisans

Bu proje MIT lisansı koşulları kapsamında lisanslanmıştır.
