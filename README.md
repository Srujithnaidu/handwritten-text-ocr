# ✍️ Handwritten Text OCR with EasyOCR and Streamlit

This project is a simple yet effective **handwritten text recognition web app** built with [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [Streamlit](https://streamlit.io/). Upload an image containing handwritten **English sentences**, and the app will extract the text and allow you to download the result.

---

## 🚀 Features

- 🔍 **OCR for handwritten text** (not printed)
- 📷 Supports `.png`, `.jpg`, and `.jpeg` image formats
- 🧠 Uses EasyOCR’s deep learning models under the hood
- 💡 Displays both original and grayscale preprocessed image
- 📥 Download recognized text as a `.txt` file
- 🌐 Runs entirely on your local machine with Streamlit

---

## 🖼 Sample

<p align="center">
  <img src="https://github.com/srujith_ecoder/handwritten-text-ocr/assets/sample_interface.png" alt="OCR Interface" width="600">
</p>

---

## 🛠️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/srujith_ecoder/handwritten-text-ocr.git
   cd handwritten-text-ocr
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit app**  
   ```bash
   streamlit run app.py

  🧾 Requirements:
    streamlit
    easyocr
    opencv-python
    numpy
    pillow
  📂 Project Structure:
  handwritten-text-ocr/
  ├── app.py                 # Main Streamlit app
  ├── requirements.txt       # Required Python packages
  └── README.md              # Project documentation

📌 Notes
  This app uses EasyOCR’s pretrained models, so it does not require any custom training or datasets.
  For accurate results, ensure the uploaded image is clean and well-scanned.
  Currently optimized for English handwritten sentences.

🔮 Future Enhancements
  ✅ Support for multiple languages
  ✅ Better image preprocessing (deskew, denoise)
  ✅ Bounding box display over detected text
  ✅ Export as PDF or DOCX


👨‍💻 Author
Made with ❤️ by Srujith{https://github.com/Srujithnaidu}
