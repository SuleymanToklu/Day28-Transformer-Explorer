# 🧠 Transformer Gözlemcisi (Transformer Explorer)

Bu proje, bir Transformer tabanlı dil modelinin içine girerek, bir cümlenin anlamını nasıl işlediğini adım adım görselleştiren interaktif bir Gradio uygulamasıdır. Kullanıcıların, modern Doğal Dil İşleme (NLP) modellerinin "zihninin" içine bakmalarını ve karmaşık anlamsal ilişkileri nasıl kurduklarını anlamalarını sağlar.

---

### Temel Özellikler

* **Model Seçimi:** `BERT`, `DistilBERT` gibi farklı Transformer tabanlı modeller arasında geçiş yapın.
* **Dikkat Deseni Görselleştirmesi:** Modelin her bir katmanındaki dikkat kafalarının (attention heads), bir cümlenin anlamını oluştururken hangi kelimelere odaklandığını ısı haritaları (heatmaps) üzerinde inceleyin.
* **Kelime Gömülmeleri Uzayı:** Kelimelerin, modelin anlamsal uzayında nasıl temsil edildiğini ve kümelendiğini PCA ile 2 boyuta indirgenmiş interaktif bir grafikte keşfedin.
* **İnteraktif Arayüz:** Tüm analizler, Gradio ile oluşturulmuş kullanıcı dostu bir web arayüzü üzerinden gerçekleştirilir.

---

### 🚀 Teknik Detaylar

Bu proje, modern makine öğrenmesi araçlarını ve kavramlarını bir araya getirir:

* **Ana Konseptler:**
    * Transformer Mimarisi
    * Self-Attention (Öz-Dikkat) Mekanizması
    * Kelime Gömülmeleri (Word Embeddings)
    * Boyut İndirgeme (PCA)

* **Kullanılan Teknolojiler:**
    * **Backend:** Python
    * **ML Kütüphaneleri:** Hugging Face `transformers`, `PyTorch`
    * **Veri Analizi:** `numpy`, `scikit-learn`
    * **Görselleştirme:** `Plotly`, `Matplotlib`, `Seaborn`
    * **Web Arayüzü & Dağıtım:** `Gradio`, Hugging Face Spaces

---

### 💻 Yerel Makinede Çalıştırma

Projeyi kendi bilgisayarınızda denemek için aşağıdaki adımları takip edebilirsiniz:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/SuleymanToklu/Day28-Transformer-Explorer.git](https://github.com/SuleymanToklu/Day28-Transformer-Explorer.git)
    cd Day28-Transformer-Explorer
    ```

2.  **Sanal Ortam Oluşturun ve Aktif Hale Getirin:**
    ```bash
    python -m venv venv
    # Windows için:
    .\venv\Scripts\activate
    # MacOS/Linux için:
    # source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı Başlatın:**
    ```bash
    python app.py
    ```

Uygulama yerel bir adreste (`http://127.0.0.1:7860`) çalışmaya başlayacaktır.
