import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import cv2

reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="Handwritten Text OCR", layout="centered")
st.title("✍️ Handwritten Text Recognition using EasyOCR")
st.write("Upload a handwritten sentence image. The app will extract the exact text and let you download it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

    with st.spinner("🔍 Extracting text..."):
        recognized = reader.readtext(gray, detail=0, paragraph=True)

    extracted_text = "\n".join(recognized)
    st.subheader("📝 Extracted Text")
    st.success(extracted_text)

    st.download_button(
        label="📥 Download Text",
        data=extracted_text,
        file_name="recognized_text.txt",
        mime="text/plain"
    )

# import streamlit as st
# import easyocr
# from PIL import Image
# import numpy as np
# import cv2

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# st.set_page_config(page_title="Handwritten Text OCR", layout="centered")
# st.title("✍️ Handwritten Text Recognition using EasyOCR")
# st.write("Upload a handwritten sentence image. The app will extract the text and let you download it.")

# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     np_image = np.array(image)
#     gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
#     st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

#     # OCR
#     with st.spinner("🔍 Recognizing handwritten text..."):
#         result = reader.readtext(gray, detail=0, paragraph=True)

#     extracted_text = "\n".join(result)
#     st.subheader("📝 Extracted Text")
#     st.success(extracted_text)

#     # Download text file
#     st.download_button(
#         label="📥 Download Text",
#         data=extracted_text,
#         file_name="recognized_text.txt",
#         mime="text/plain"
#     )

# import streamlit as st
# import easyocr
# from PIL import Image
# import numpy as np

# # Streamlit settings
# st.set_page_config(page_title="OCR with EasyOCR", layout="centered")
# st.title("✍️ Handwritten Text OCR App")
# st.write("Upload an image with handwritten text to extract content using EasyOCR.")

# # Upload image
# uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

#     # Convert image to numpy array
#     image_np = np.array(image)

#     # EasyOCR Reader
#     reader = easyocr.Reader(['en'])

#     with st.spinner("🔍 Extracting Text..."):
#         results = reader.readtext(image_np, detail=0)
#         text = "\n".join(results)

#     # Display results
#     st.subheader("📜 Extracted Text")
#     st.code(text)

#     # Download extracted text
#     st.download_button("⬇ Download as TXT", data=text, file_name="output.txt")

# import easyocr
# import streamlit as st
# from PIL import Image
# import numpy as np
# import io

# # Setup EasyOCR
# reader = easyocr.Reader(['en'])

# st.title("📝 Handwritten Text Recognition")
# st.write("Upload a handwritten image to extract text.")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Show uploaded image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert to grayscale (optional)
#     gray = np.array(image.convert("L"))
#     st.image(gray, caption="Preprocessed (Grayscale)", use_column_width=True)

#     # OCR
#     with st.spinner("🔍 Extracting text..."):
#         result = reader.readtext(gray, detail=0)
#         extracted_text = "\n".join(result)

#     # Show text
#     st.subheader("📜 Extracted Text")
#     st.code(extracted_text)

#     # Download
#     st.download_button("📥 Download Extracted Text", data=extracted_text, file_name="extracted_text.txt")

# import streamlit as st
# from PIL import Image
# import easyocr
# import numpy as np
# import cv2
# import io

# # Streamlit UI setup
# st.set_page_config(page_title="Handwritten Text OCR", layout="centered")
# st.title("📝 OCR for Handwritten Text")
# st.write("Upload a handwritten image to extract and download the text.")

# # OCR reader
# reader = easyocr.Reader(['en'])

# # Upload file
# uploaded_file = st.file_uploader("Upload Handwritten Image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Load image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image for better clarity
#     img_np = np.array(image)
#     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#     st.subheader("🔍 Grayscale Preprocessed Image")
#     st.image(gray, use_column_width=True, clamp=True, caption="Grayscale Image")

#     # OCR
#     with st.spinner("Extracting text..."):
#         result = reader.readtext(gray, detail=0)
#         extracted_text = "\n".join(result)

#     # Show result
#     st.subheader("🔠 Extracted Text:")
#     st.code(extracted_text)

#     # Download as .txt file
#     st.download_button(
#         label="📥 Download Extracted Text",
#         data=extracted_text,
#         file_name="extracted_text.txt",
#         mime="text/plain"
#     )

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from torchvision import transforms
# from model import CNNModel
# from dataset import HandwrittenCharactersDataset

# # Load model
# model = CNNModel(num_classes=671)  # ✅ change to match your dataset
# model.load_state_dict(torch.load("handwritten_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # Streamlit UI
# st.set_page_config(page_title="OCR for Handwritten Text", layout="centered")
# st.title("📝 OCR for Handwritten Text")
# st.write("Upload a handwritten character image to recognize it and download the prediction.")

# # Preprocessing
# def preprocess_image(pil_image):
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
#     return gray

# # Upload
# uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     processed = preprocess_image(image)
#     st.image(processed, caption="🛠️ Grayscale & Resized", use_column_width=True, clamp=True)

#     # To tensor
#     transform = transforms.Compose([transforms.ToTensor()])
#     input_tensor = transform(Image.fromarray(processed)).unsqueeze(0)

#     # Predict
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_idx = torch.argmax(output, 1).item()

#     # Get label
#     label_map = {v: k for k, v in HandwrittenCharactersDataset("dataset/data").label_to_idx.items()}
#     predicted_label = label_map.get(predicted_idx, "Unknown")

#     # Show result
#     st.subheader("🔠 Predicted Character:")
#     st.success(f"Model Prediction: **{predicted_label}**")

#     # Download result
#     st.download_button(
#         label="📥 Download Result as Text",
#         data=predicted_label,
#         file_name="prediction.txt",
#         mime="text/plain"
#     )
# --------------------------------------------------------------------------------------
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from torchvision import transforms
# from model import CNNModel
# from dataset import HandwrittenCharactersDataset

# # Load model
# data_path = "dataset/data"
# temp_dataset = HandwrittenCharactersDataset(data_path)
# model = CNNModel(num_classes=len(temp_dataset.label_to_idx))
# model.load_state_dict(torch.load("handwritten_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # Streamlit page setup
# st.set_page_config(page_title="OCR for Handwritten Text", layout="centered")
# st.title("📝 OCR for Handwritten Text")
# st.write("Upload a handwritten image to predict the handwritten character.")

# # Preprocessing function
# def preprocess_image(pil_image):
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)  # Match training size
#     return gray

# # File uploader
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     processed = preprocess_image(image)
#     st.subheader("🛠️ Preprocessed Image")
#     st.image(processed, caption="Grayscale & Resized", use_column_width=True, clamp=True)

#     # Convert to tensor
#     # transform = transforms.Compose([
#     #     transforms.ToTensor()
#     # ])
#     # input_tensor = transform(Image.fromarray(processed)).unsqueeze(0)

#     # # Predict
#     # with torch.no_grad():
#     #     output = model(input_tensor)
#     #     predicted_idx = torch.argmax(output, 1).item()
#         # Convert to tensor
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     input_tensor = transform(Image.fromarray(processed)).unsqueeze(0)

#     # Predict
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_idx = torch.argmax(output, 1).item()

#     # Get label
#     label_map = {v: k for k, v in HandwrittenCharactersDataset("dataset/data").label_to_idx.items()}
#     predicted_label = label_map.get(predicted_idx, "Unknown")

#     # Display result
#     st.subheader("🔠 Predicted Label:")
#     st.success(f"Model Prediction: **{predicted_label}**")

#     # Optional: Let user download prediction
#     st.download_button("📥 Download Prediction", data=predicted_label, file_name="prediction.txt")
# -------------------------------------------------------------------------------------------------------------------------
    # Get label
#     label_map = {v: k for k, v in temp_dataset.label_to_idx.items()}
#     predicted_label = label_map.get(predicted_idx, "Unknown")

#     # Display result
#     st.subheader("🔠 Predicted Label:")
#     st.success(f"Model Prediction: **{predicted_label}**")

#     # Create a list to hold predictions if uploading multiple images in the future
# predicted_characters = []

# # Predict the label
# with torch.no_grad():
#     output = model(input_tensor)
#     predicted_idx = torch.argmax(output, 1).item()

# # Get label mapping from dataset
# label_map = {v: k for k, v in HandwrittenCharactersDataset("dataset/data").label_to_idx.items()}
# predicted_label = label_map.get(predicted_idx, "Unknown")

# # Add to predictions list
# predicted_characters.append(predicted_label)

# # Join characters into string
# predicted_text = ''.join(predicted_characters)

# # Display result
# st.subheader("🔠 Predicted Text:")
# st.success(predicted_text)

# # Enable download of result
# st.download_button(
#     label="📥 Download Predicted Text",
#     data=predicted_text,
#     file_name="predicted_text.txt",
#     mime="text/plain"
# )

# -----------------------------------------------------------------------------------------

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from torchvision import transforms
# from model import CNNModel
# from dataset import HandwrittenCharactersDataset

# # Load model
# model = CNNModel(num_classes=26)
# model.load_state_dict(torch.load("handwritten_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # Streamlit page setup
# st.set_page_config(page_title="OCR for Handwritten Text", layout="centered")
# st.title("📝 OCR for Handwritten Text")
# st.write("Upload a handwritten image to predict the handwritten character.")

# # Preprocessing function
# def preprocess_image(pil_image):
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_LINEAR)
#     return gray

# # File uploader
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     processed = preprocess_image(image)
#     st.subheader("🛠️ Preprocessed Image")
#     st.image(processed, caption="Grayscale & Resized", use_column_width=True, clamp=True)

#     # Convert to tensor
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     input_tensor = transform(Image.fromarray(processed)).unsqueeze(0)

#     # Predict
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_idx = torch.argmax(output, 1).item()

#     # Get label
#     label_map = {v: k for k, v in HandwrittenCharactersDataset("dataset/data").label_to_idx.items()}
#     predicted_label = label_map.get(predicted_idx, "Unknown")

#     # Display result
#     st.subheader("🔠 Predicted Character:")
#     st.success(f"Model Prediction: **{predicted_label}**")

# ---------------------------------------------------------------


# from model import CNNModel

# import streamlit as st
# import easyocr
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from dataset import HandwrittenCharactersDataset

# # Streamlit page setup
# import torch

# # Load model
# model = CNNModel(num_classes=26)  # Assuming 26 classes (A–Z)
# model.load_state_dict(torch.load("handwritten_model.pth", map_location=torch.device('cpu')))
# model.eval()

# st.set_page_config(page_title="OCR for Handwritten Text", layout="centered")
# st.title("📝 OCR for Handwritten Text")
# st.write("Upload a handwritten image to extract and digitize its content.")

# # Image preprocessing function
# def preprocess_image(pil_image):
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary

# # File uploader
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     preprocessed_image = preprocess_image(image)
#     st.subheader("🛠️ Preprocessed Image")
#     st.image(preprocessed_image, caption="For OCR", use_column_width=True, clamp=True)

#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
#     result = reader.readtext(preprocessed_image)

#     if result:
#         extracted_text = "\n".join([text for _, text, _ in result])
#         st.success(f"✅ Detected {len(result)} text regions.")
#     else:
#         extracted_text = "⚠️ No text detected. Try uploading a clearer image."
#         st.warning("No text found in the image.")

#     st.subheader("📄 Extracted Text:")
#     st.text_area("Detected Text", extracted_text, height=200)

#     st.download_button("📥 Download Text File", extracted_text, file_name="extracted_text.txt")
