import math
import os
import smtplib
import ssl
import tkinter as tk
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import dotenv
import numpy as np
import PIL.Image
import PIL.ImageTk
from deepface import DeepFace
from keras.applications.resnet_v2 import preprocess_input
from keras.models import load_model
from PIL import (
    Image,
    ImageTk,
)

dotenv.load_dotenv()

EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", None)
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", None)

if EMAIL_USERNAME is None or EMAIL_PASSWORD is None:
    raise ValueError(
        "You should set EMAIL_USERNAME and EMAIL_PASSWORD environment variables to run this app"
    )


basedir = os.path.abspath(os.path.dirname(__file__))
# face_classifier_path="haarcascade_frontalface_default.xml"
face_classifier_path = os.path.join(
    basedir, "models", "haarcascade_frontalface_default.xml"
)
# person_analyzer_model_path = "person_analyzer.h5"
person_analyzer_model_path = os.path.join(basedir, "models", "person_analyzer.h5")
# face_shape_model_path="face_shape_model.h5"
face_shape_model_path = os.path.join(basedir, "models", "face_shape_model.h5")

if not os.path.exists(face_classifier_path):
    raise ValueError(f"face_classifier doesn't exists: {face_classifier_path}")

if not os.path.exists(person_analyzer_model_path):
    raise ValueError(
        f"person_analyzer_model doesn't exists: {person_analyzer_model_path}"
    )

if not os.path.exists(face_shape_model_path):
    raise ValueError(f"face_shape_model doesn't exists: {face_shape_model_path}")

face_classifier = cv2.CascadeClassifier(face_classifier_path)
person_analyzer = load_model(person_analyzer_model_path)
face_shape_analyzer = load_model(face_shape_model_path)

# Face Shapes
labels_of_personalities = ["İyi biri", "Kötü biri"]
labels_face_shapes = ["Kalp", "Dikdörtgen", "Oval", "Yuvarlak", "Kare"]


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Stream with Rectangle Drawing")

        self.video_source = 0  # Use the default camera (change as needed)
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(3), height=800)
        self.canvas.pack(side=tk.LEFT)

        self.image_canvas = tk.Canvas(root, width=640, height=480)
        self.image_canvas.pack(side=tk.RIGHT)
        self.image_label = None

        # LABELS IN THE SCREEN
        self.rectangle = None
        self.label1_ = tk.Label(root, text="")
        self.label2_ = tk.Label(root, text="")
        self.label3_ = tk.Label(root, text="")
        self.label4_ = tk.Label(root, text="")
        self.label5_ = tk.Label(root, text="")
        self.label6_ = tk.Label(root, text="Mail")
        self.entry = tk.Entry(root, font=("Arial", 16))
        self.label7_ = tk.Label(root, text="")

        self.answer_to_mail = False

        self.label1_.config(font=("Arial", 18))
        self.label1_.config(fg="black")
        self.label1_.pack()

        self.label2_.config(font=("Arial", 18))
        self.label2_.config(fg="black")
        self.label2_.pack()

        self.label3_.config(font=("Arial", 18))
        self.label3_.config(fg="black")
        self.label3_.pack()

        self.label4_.config(font=("Arial", 18))
        self.label4_.config(fg="black")
        self.label4_.pack()

        self.label5_.config(font=("Arial", 18))
        self.label5_.config(fg="black")
        self.label5_.pack()

        self.label6_.config(font=("Arial", 18))
        self.label6_.config(fg="black")
        self.label6_.pack()

        self.label7_.config(font=("Arial", 18))
        self.label7_.config(fg="black")
        self.label7_.pack()

        self.entry.pack()

        self.is_streaming = True

        self.is_processing = False

        self.canv_counter = 0  # this number provide to run part of code once

        self.goodGuyCounter = 0
        self.badGuyCounter = 0
        self.labelDic = {
            "Kalp": 0,
            "Dikdörtgen": 0,
            "Oval": 0,
            "Yuvarlak": 0,
            "Kare": 0,
        }

        self.update()
        self.start_video()

    def start_video(self):
        self.is_streaming = True

    def process_video(self):
        ret, self.frame = self.vid.read()
        if self.is_processing:
            database_path = "celebrities_images"

            face = self.frame[100:400, 200:450]

            # TO FIND MOST SIMILAR FACE WITH
            dfs = DeepFace.find(
                img_path=self.frame,
                db_path=database_path,
                enforce_detection=False,
            )

            # These try expcepts catch the errors caused from not detecting face
            try:
                self.celebrity_pic_path = dfs[0].loc[0]["identity"]
                print(dfs[0])
            except Exception as e:
                print("Yüz bulunamadı", e)
                return

            try:
                self.celebrity_name = (
                    dfs[0].loc[0]["identity"].split("\\")[-1].split("/")[0]
                )
                celebrity_pic_name = (
                    dfs[0].loc[0]["identity"].split("\\")[-1].split("/")[1]
                )
                analyze1 = None

                analyze1 = DeepFace.analyze(face, actions=("age", "gender", "race"))
            except Exception as e:
                print("Yüz bulunamadı", e)
                return

            for i in range(len(dfs[0])):
                try:
                    self.celebrity_pic_path = dfs[0].loc[i]["identity"]
                    analyze2 = DeepFace.analyze(
                        cv2.imread(self.celebrity_pic_path),
                        actions=("age", "gender", "race"),
                    )
                except Exception as e:
                    print("YÜZ BULUNAMADI", e)
                    return

                if analyze1[0]["dominant_gender"] == analyze2[0]["dominant_gender"]:
                    self.celebrity_name = (
                        dfs[0].loc[0]["identity"].split("\\")[-1].split("/")[0]
                    )
                    break
                else:
                    pass
            if not self.is_processing:
                print("yüz tespit edilemedi")
                return

            labels = []
            labels2 = []

            similarity_rate = (
                (math.pi - math.acos(dfs[0].loc[0]["VGG-Face_cosine"])) * 100 / math.pi
            )

            initial_image = Image.open(self.celebrity_pic_path)
            initial_image = initial_image.resize((640, 480))
            self.tk_image = ImageTk.PhotoImage(initial_image)

            if self.canv_counter < 1:
                self.image_label = tk.Label(self.root, image=self.tk_image)
                self.image_label.place(x=640, y=0)
                self.canv_counter += 1
            else:
                self.image_label.config(image=self.tk_image)

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # barisin kodu
            self.frame_center_x = self.frame.shape[1] // 2
            self.frame_center_y = self.frame.shape[0] // 2

            closest_face = None
            min_distance = float("inf")  ##BU NE ANLAMADIM

            is_this_first = True
            for x, y, w, h in faces:
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                distance = (
                    (self.frame_center_x - face_center_x) ** 2
                    + (self.frame_center_y - face_center_y) ** 2
                ) ** 0.5

                if distance < min_distance and is_this_first:
                    min_distance = distance
                    closest_face = (x, y, w, h)
                    is_this_first = False

                x, y, w, h = closest_face
                roi_gray = gray[y : y + h, x : x + w]

                # Resize the input to match MobileNetV2's requirements
                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                roi_resized = cv2.resize(roi_rgb, (224, 224))
                # Preprocess the input
                roi_preprocessed = preprocess_input(roi_resized)
                roi_preprocessed = cv2.resize(roi_rgb, (200, 200))

                # each code for second model
                roi_gray2 = gray[y : y + h, x : x + w]
                roi_rgb2 = cv2.cvtColor(roi_gray2, cv2.COLOR_GRAY2RGB)
                roi_resized2 = cv2.resize(roi_rgb2, (224, 224))
                roi_preprocessed2 = preprocess_input(roi_resized2)
                roi_preprocessed2 = cv2.resize(roi_rgb2, (150, 150))

                if np.sum([roi_preprocessed]) != 0 and np.sum([roi_preprocessed2]) != 0:
                    # Expand the dimensions to match the model's input shape
                    roi_preprocessed = np.expand_dims(roi_preprocessed, axis=0)
                    roi_preprocessed2 = np.expand_dims(roi_preprocessed2, axis=0)
                    preds = person_analyzer.predict(roi_preprocessed)[0]
                    preds2 = face_shape_analyzer.predict(roi_preprocessed2)[0]

                    self.label1 = labels_of_personalities[preds.argmax()]
                    self.label2 = labels_face_shapes[preds2.argmax()]

                    if self.label1 == "İyi biri":
                        self.goodGuyCounter += 1
                    else:
                        self.badGuyCounter += 1
                        self.labelDic[self.label2] += 1

                    if self.goodGuyCounter > self.badGuyCounter:
                        firstVal = "İyi biri"
                        self.res = (
                            self.goodGuyCounter
                            / (self.badGuyCounter + self.goodGuyCounter)
                            * 100
                        )
                    else:
                        firstVal = "Kötü biri" # Bunu nerede kullanıyorsun?
                        self.res = (
                            self.badGuyCounter
                            / (self.badGuyCounter + self.goodGuyCounter)
                            * 100
                        )

                    self.labelDic[self.label2] += 1

                    self.res2 = max(self.labelDic, key=lambda k: self.labelDic[k])
                    self.res3 = (
                        max(self.labelDic.values())
                        / (sum(self.labelDic.values()))
                        * 100
                    )

                    self.label1_.config(
                        text="İyilerin sayacı: {}".format(self.goodGuyCounter)
                    )
                    self.label2_.config(
                        text="Kötülerin sayacı: {}".format(self.badGuyCounter)
                    )
                    self.label3_.config(text="Yüz şekli: {}".format(self.label2))
                    self.label4_.config(
                        text="{}\nYaşınız: {}\nKökeniniz: {}".format(
                            self.detect_personality_short(
                                self.label1, self.label2, self.res, self.res3
                            ),
                            analyze1[0]["age"] - 5,
                            analyze1[0]["dominant_race"],
                        )
                    )
                    self.label5_.config(
                        text=f"{self.celebrity_name}\nBenzerlik oranı %{similarity_rate:.2f}"
                    )

                    self.label1_.place(x=10, y=500)
                    self.label2_.place(x=10, y=535)
                    self.label3_.place(x=10, y=570)
                    self.label4_.place(x=300, y=530)
                    self.label5_.place(x=800, y=500)
                    self.label6_.place(x=10, y=610)
                    self.entry.place(x=60, y=610)

                    self.is_processing = False
                else:
                    # There is no face in the screen
                    pass

            # To place text on video, we define image
        pil_image = Image.fromarray(
            cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        )

        self.photo = PIL.ImageTk.PhotoImage(image=pil_image)

    def change_process_state(self, event):
        if not self.is_processing:
            self.is_processing = True

    def update(self):
        ret, self.frame = self.vid.read()

        if self.is_streaming and ret:
            self.process_video()

            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        # ****************************

        root.bind("<KeyPress-T>", self.change_process_state)
        root.bind("<Return>", self.yes_to_send_mail)

        if self.rectangle:
            self.canvas.delete(self.rectangle)
            self.rectangle = None

        if self.is_streaming:
            self.rectangle = self.canvas.create_rectangle(
                200, 100, 450, 400, outline="red", width=6
            )

        self.root.after(10, self.update)

    def yes_to_send_mail(self, event):
        if self.entry.get() != "":
            self.answer_to_mail = True
            self.alici = self.entry.get()
            self.entry.delete(0, tk.END)
            try:
                self.send_mail(self.label1, self.res2, self.res, self.res3, self.alici)
            except Exception as e:
                print("Bir hata oluştu lütfen tekrar deneyin", e)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def send_mail(self, firstVal, res2, res, res3, alici):
        # use input()
        # alici = input('Mail adresinizi giriniz: ')
        baslik = "Hacettepe Yapay Zeka Topluluğu"
        mesaj = self.detect_personality(firstVal, res2, res, res3)
        mesaj = mesaj + f"\nSana en çok benzeyen kişi: {self.celebrity_name}\n"
        context = ssl.create_default_context()

        posta = MIMEMultipart()
        posta["From"] = EMAIL_USERNAME
        posta["To"] = alici
        posta["Subject"] = baslik

        with open(self.celebrity_pic_path, "rb") as image_file:
            image_data = image_file.read()
            image = MIMEImage(image_data, name="image.jpg")
            posta.attach(image)

        posta.attach(MIMEText(mesaj, "plain"))
        # Convert the message to a string
        posta_str = posta.as_string()
        port = 465
        host = "smtp.gmail.com"
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host=host, port=port, context=context) as epostaSunucusu:
            epostaSunucusu.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            epostaSunucusu.sendmail(EMAIL_USERNAME, alici, posta_str)

        print("Mail başarıyla gönderildi")

    def detect_personality(self, firstVal, res2, res, res3):
        if firstVal == "İyi biri" and res2 == "Kalp":
            mesaj = f"""\
        İyi biri olma olasılığı {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        İyi niyetli ve kalp yüzlü bir kişi:
        Kalp şeklinde bir yüze sahip bir kişi, "iyi niyetli" olarak tanımlandığında sıcak, içten ve şefkatli olarak görülebilir. İşte kalp şeklinde yüzü olan bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

        Empatik: Kalp şeklinde bir yüz, başkalarının duygularına ve ihtiyaçlarına hassas ve duyarlı bir doğayı yansıtabilir. Diğer insanların duygularına karşı duyarlı biri olarak algılanabilirler.

        Dostça: Kalp şeklindeki yüzler, yaklaşılabilir ve arkadaş canlısı olarak görünebilir, bu da onları başkalarıyla kolayca iletişim kurulabilen kişiler yapar.

        Romantik: Kalp şeklindeki yüz, romantizm ve sevgi dolu bir tavır ile ilişkilendirilebilir, duygusal bağlantıları ve ilişkileri önemseyen bir kişiyi ima edebilir.

        Duyarlı: Kalp şeklinde yüzü olan kişiler, çevrelerindeki insanların duygularına karşı hassas oldukları şeklinde algılanabilirler ve duygusal zeka seviyeleri yüksek olabilir.

        Hayırsever: Kalp şeklinde yüze sahip bir kişi, başkalarına yardım etmeyi ve topluma geri vermeği seven biri olarak görülebilir.

        Olumlu: Yüz yapısı genel olarak olumlu ve iyimser bir tavır katkıda bulunabilir.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
            """
        elif firstVal == "İyi biri" and res2 == "Dikdörtgen":
            mesaj = f"""\
        İyi biri olma olasılığı {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Dikdörtgen yüz şekli olan bir kişi, "iyi niyetli" olarak tanımlandığında zeki, yaklaşılabilir ve güvenilir olarak algılanabilir. İşte bir iyi insan imajına sahip biri ve Dikdörtgen yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

        Zeka: Dikdörtgen yüz şekli genellikle zeka izlenimi yaratır ve bireyin düşünceli ve analitik olduğunu düşündürebilir.

        Yaklaşılabilir: Dikdörtgen yüze sahip insanlar, yaklaşılabilir ve dostça görünebilir, bu da diğer insanların onlarla kolayca iletişim kurmasını sağlar.

        Güvenilirlik: Dikdörtgen yüz, güvenilirlik ve güvenilirlik hissi ile ilişkilendirilebilir, bu da bir "iyi niyetli" olmakla uyumlu bir özelliktir.

        Sakin ve Toparlanmış: Dikdörtgen yüzler sakin ve toparlanmış bir görünüm sergileyebilir, bu da kişinin zorlu durumları zarafetle ele alabileceğini gösterir.

        Çalışkan: Bu yüz şekli, çalışkan ve işine sadık bir kişilikle ilişkilendirilebilir, sorumluluklarına bağlı biri olarak tanımlanabilirler.

        Sorumlu: Dikdörtgen yüze sahip bireyler, sorumlu ve düzenli olarak algılanabilirler, bu da "iyi niyetli" imajlarına katkıda bulunur.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "İyi biri" and res2 == "Oval":
            mesaj = f"""\
        İyi biri olma olasılığı {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Oval yüz şekli olan bir kişi, "iyi niyetli" olarak tanımlandığında dengeli, dostça ve yaklaşılabilir olarak algılanabilir. Bir iyi niyetli imajına ve oval yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

        Dostça: Oval yüzler genellikle yaklaşılabilir ve dostça olarak görülür, bu da diğer insanların onlarla kolayca bağlantı kurabilmesini sağlar.

        Dengeli: Oval yüz şekli dengeli ve uyumlu olarak kabul edilir, bu da dengeli ve sakin bir kişiliği yansıtabilir.

        Sosyal: Oval yüzlü bireyler sosyal ve dışa dönük olarak algılanabilirler, başkalarıyla etkileşimden keyif alan biri olarak tanımlanabilirler.

        Empatik: Oval yüzlü insanlar empatik ve anlayışlı olarak görülebilirler, çevrelerindekilerin duygularına önem veren bir tutum sergileyebilirler.

        Rahat: Oval bir yüz, rahatlık ve uyum sağlama hissi ile iletebilir, bu da rahat ve uyumlu bir kişi olduğunu düşündürebilir.

        Diplomatik: Oval yüz şekline sahip bireyler iletişim ve çatışma çözme konusunda başarılı olabilirler, bu da bir "iyi niyetli" imajıyla uyumlu bir şekilde uyumu teşvik eden bir kişiyi yansıtabilir.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "İyi biri" and res2 == "Yuvarlak":
            mesaj = f"""\
        İyi biri olma olasılığı {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Yuvarlak yüz şekli olan bir kişi, "iyi insan" olarak tanımlandığında dostça, yaklaşılabilir ve neşeli olarak algılanabilir. İşte bir iyi insan imajına sahip biri ve yuvarlak yüze sahip bir kişiye atfedilebilecek bazı özellikler ve ilişkilendirmeler:

        Dostça: Yuvarlak yüzler genellikle sıcak ve dostça bir tavır ile iletilir, bu da diğer insanların onların etrafında rahat hissetmelerini kolaylaştırır.

        Yaklaşılabilir: Yuvarlak yüzlü insanlar yaklaşılabilir ve açık olarak görünebilir, diğerlerini kendileriyle iletişime davet ederler.

        Sosyal: Yuvarlak yüzlü bireyler sosyal ve dışa dönük olarak algılanabilirler, etkileşimleri ve ilişkileri keyifli bulabilirler.

        İyiliksever: Yuvarlak bir yüz şekli nazik ve iyi niyetli bir doğayı ima edebilir, bakım ve düşünceli olma imajıyla uyum sağlar.

        Mizahi: Bazıları yuvarlak bir yüzü mizah anlayışıyla ilişkilendirebilir, bu da kişinin gülmeyi ve neşeli olmayı sevdiğini ima edebilir.

        Olumlu Tutum: Yüz yapısı, hayata olumlu ve iyimser bir bakış açısı ile iletebilir.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "İyi biri" and res2 == "Kare":
            mesaj = f"""\
        İyi biri olma olasılığı {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Kare yüz şekline sahip bir kişi genellikle güçlü, kendine güvenen ve kararlı olarak algılanır. "İyi insan" tanımı ile birleştiğinde, bu kişinin karakterinde bu olumlu özellikleri yansıttığını düşündürebilir. İşte bir "iyi insan" imajına ve kare yüze sahip bir kişiyle ilişkilendirebileceğiniz bazı ifadeler veya özellikler:

        Güvenilir: Kare yüzlü insanlar genellikle güvenilirlik ve güvenilirlik izlenimi verir, bu da bir "iyi insan" olma fikriyle uyumlu bir özelliktir.

        Güçlü Çene: Kare yüzler genellikle güçlü bir çeneyi simgeler, bu da kararlılık ve dayanıklılığı temsil eder.

        Dürüst Görünüm: Kare yüz dürüstlüğü ve açıklığı ile ilişkilendirilebilir, bu da bir "iyi insan" fikrini pekiştirir.

        Liderlik Potansiyeli: Kare yüzlü bireyler, yüz yapıları özgüven ve kararlılık gösterebileceğinden doğal liderler olarak görülebilirler.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "Kötü biri" and res2 == "Kalp":
            mesaj = f"""\
        Probability of being Kötü biri is {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Kalp şeklinde bir yüze sahip bir "harikulade olmayan insan" karakterini tanımlarken, yüz hatlarının karakterlerine karmaşıklık kattığını söyleyebiliriz.

        Onun kalp şeklindeki yüzünün yumuşak ve çekici hatlarına rağmen, onun hakkında ürkütücü bir şeyler var. Görünüşte masum özellikleri, içinde gizlenen kötülükle sert bir tezat oluşturuyor.
        Sanki aldatıcı tatlı görünümünü, daha karanlık bir ajandayı gizlemek için bir maskara olarak kullanıyormuş gibi. Yüzünün nazik eğrilerine aldanmayın; bu görünüşün altında kurnaz ve hain bir zeka var.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "Kötü biri" and res2 == "Dikdörtgen":
            mesaj = f"""\
        Probability of being Kötü biri is {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Dikdörtgen yüzlü harikulade olmayan bir insanın yüz hatları uzun ve potansiyel olarak tehditkar görünebilir.

        Onun dikdörtgen yüzü, genel görünümüne ürkütücü bir kalite katıyor. Uzun yüz hatları, tehditkar varlığını daha da artırıyor ve neredeyse başka bir dünyevi bir hava kazandırıyor.
        Keskin, dar hatlara ve gözlerindeki tehditkar parıltıya sahip, manipülasyon ve aldatma konusunda zevk alan kötü adam tipini anımsatıyor. 
        Dikdörtgen yüzü, her zaman kötü niyetli planlarında bir adım önde gibi göründüğü endişe duygusunu artırıyor gibi, sanki her zaman bir adım önde gibi.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "Kötü biri" and res2 == "Oval":
            mesaj = f"""\
        Probability of being Kötü biri is {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Oval yüzlü harikulade olmayan bir insan, daha esrarlı ve sofistike bir görünüme sahip olabilir.

        Oval yüzü, hesaplanmış kötülüğün bir havasını yayar. Vücudu düzgün, simetrik hatları, gerilimi arttırmak için gerçek niyetini rahatsız edici bir cazibe ile örtüyor gibi görünüyor. 
        Özellikleri, geleneksel olarak çekici olsa da, gizli bir ajandayı ima eden anlaşılmaz bir kalite taşıyor. Keskin bir bakış ve hesaplanmış ve zarif bir tavır ile, gölgede iş yapan, çekici oval yüzünü kullanarak karanlık ve sofistike doğasını gizlemek için kullanan harikulade olmayan bir karakter tipidir.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "Kötü biri" and res2 == "Yuvarlak":
            mesaj = f"""\
        Probability of being Kötü biri is {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Yuvarlak yüzlü bir kötü karakter, insanları yanıltıcı bir görünüme sahip olabilir ve insanları yanıltıcı bir güven duygusuna sokabilir. İşte bir tanım:

        Onun yuvarlak yüzü, görünüşte dostça cephesinin altında gizli bir kurnaz doğayı gizliyor. En dikkatli gözlemcileri bile etkisiz hale getirebilecek yumuşak, kavisli özelliklerle, kötülüğünü maskelemekte usta.
        Yuvarlak yüzü, topluma sorunsuz bir şekilde karışmasına izin verirken, görünüşte masum tavırları yüzeyin altında pusuya düşen kötü niyetleri ele vermez. O, alçakgönüllü görünüşünü kullanarak manipüle etmek ve aldatmak için kullanan  harikulade olmayan bir insandır, bu da onu daha da tehlikeli kılar.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        elif firstVal == "Kötü biri" and res2 == "Kare":
            mesaj = f"""\
        Probability of being Kötü biri is {res}
        Yüz şeklinin {res2} olma olasılığı {res3}

        Kare yüz şekli, harikulade olmayan bir insanın karakteri ile ilişkilendirildiğinde, etkileyici ve potansiyel olarak tehditkar bir görünümlerini vurgulayabilir.

        Onun kare yüzü, hakimiyet ve tehditkarlık havası yayar. Keskin, oyma gibi köşeleri ve güçlü çene hattı ile, özellikleri göz ardı edilmesi zor bir hakim varlığı yansıtıyor gibi görünüyor.
        Bu kare yüzlü kişi, etkileyici ve tehditkar bir kişilik taşıyormuş gibi görünüyor, onu yakından takip etmek isteyeceğiniz bir karakter olabilir.


        Sosyal Medya Hesaplarımız:
        https://linktr.ee/hacettepeaiclub
        """
        return mesaj

    def detect_personality_short(self, firstVal, res2, res, res3):
        if firstVal == "İyi biri" and res2 == "Kalp":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Bu kişi, içtenlikle dolu ve iyiliksever bir ruha sahiptir.
Kalp şeklindeki yüzü, onun sevgi dolu ve duygusal
bir insan olduğunu gösteriyor."""

        elif firstVal == "İyi biri" and res2 == "Dikdörtgen":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Dikdörtgen yüz şekli olan bir kişi, 
Bu kişi, hem iyi niyetli hem de son derece zeki
bir birey olarak öne çıkar. Ayrıca, yaklaşılabilir kişiliği ve
güvenilirliği sayesinde çevresindeki insanlar tarafından rahatlıkla
yaklaşılabilir ve güvenilir bulunur."""

        elif firstVal == "İyi biri" and res2 == "Oval":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Oval yüz şekli olan bir kişi, 
Bu kişi,coğu kişiye karşı "iyi niyetli" bir tavır sergiler.
Aynı zamanda dostça ve yaklaşılabilir kişiliği
sayesinde çevresindeki insanlar tarafından olumlu
bir şekilde algılanır."""

        elif firstVal == "İyi biri" and res2 == "Yuvarlak":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Bu kişi, yuvarlak yüzlü tatlı bir görünüme sahiptir,
ancak bu dış görünüşünün ardında insanları yanıltıcı bir kişilik gizler.
Bu kişi, insanları yanıltıcı bir güven
duygusuna sokarak onları aldatmaya çalışabilir."""

        elif firstVal == "İyi biri" and res2 == "Kare":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Kare yüz şekline sahip olan bir kişi genellikle güçlü,
kendine güvenen ve kararlı bir izlenim bırakır.
Bu yüz şekli, kişinin güçlü bir karaktere ve
kararlılığa sahip olduğunu işaret eder.."""

        elif firstVal == "Kötü biri" and res2 == "Kalp":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Kalp şeklinde bir yüze sahip bir 
"harikulade olmayan insan" karakterini tanımlarken,
yüz hatlarının karakterlerine karmaşıklık kattığını
söyleyebiliriz."""

        elif firstVal == "Kötü biri" and res2 == "Dikdörtgen":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Dikdörtgen yüzlü bir kişi, genellikle yüz hatları uzun ve
potansiyel olarak tehditkar bir izlenim verebilir.
Ancak bu, kişinin kötü biri olduğu anlamına gelmez,
çünkü insanların kişilikleri ve içsel nitelikleri yüz
hatlarından daha fazlasını ifade eder."""

        elif firstVal == "Kötü biri" and res2 == "Oval":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Bu kişi, genellikle daha esrarlı ve sofistike bir görünüme sahip olabilir.
Bu yüz şekli, yuvarlak ve oval hatlarıyla dikkat çeker ve
bu da kişiye zarif bir hava katar.
Kişinin yüz şekli kişinin duruşunu ve çekiciliğini vurgulayabilir,
esrarlı ve sofistike bir izlenim bırakabilir."""

        elif firstVal == "Kötü biri" and res2 == "Yuvarlak":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Yuvarlak yüzlü bir kötü karakter, dış görünüşünün arkasında
insanları yanıltıcı bir kişilik barındırabilir.
Bu kişi, masum veya samimi bir izlenim yaratarak
insanları yanıltıcı bir güven duygusuna sokmaya çalışabilir."""

        elif firstVal == "Kötü biri" and res2 == "Kare":
            mesaj = f"""\
İyi biri olma olasılığı {res}
Yüz şeklinin {res2} olma olasılığı {res3}

Bu yüz şekli, sivri hatları ve güçlü çene yapısıyla dikkat çeker,
bu da kişinin kararlılık ve belirginlik yansıtabilir.
Ancak bu, kişinin karakterinin tamamını tanımlamaz,
çünkü içsel nitelikler ve davranışlar yüz hatlarından
daha fazlasını ifade eder."""
        return mesaj


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
