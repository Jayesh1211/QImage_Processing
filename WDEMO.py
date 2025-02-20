import streamlit as st
import numpy as np
import cv2
import random
import math
from scipy import linalg

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def rescale_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

def tent(x, d):
    if 0 <= x < d:
        return x/d
    elif d <= x < 1:
        return (1-x)/(1-d)
    return 0

def decimalToBinary(n):
    return "{0:02b}".format(int(n))

def NOT(x):
    return "1" if x == "0" else "0"

def CNOT(x, y):
    return NOT(y) if x == "1" else y

def process_watermark(watermark_gray, cover_gray_256):
    # SVD Process
    A = watermark_gray.astype(float)
    I = cover_gray_256.astype(float)
    
    Uw, Sw, Vw = linalg.svd(A, full_matrices=True)
    Uc, Sc, Vc = linalg.svd(I, full_matrices=True)
    
    k = (256, 256)
    Swm = np.zeros(k)
    Scm = np.zeros(k)
    
    for i in range(min(256, len(Sw))):
        Swm[i,i] = Sw[i]
        Scm[i,i] = Sc[i]
    
    Awa = np.dot(Uw, Swm)
    
    Scmp = np.zeros(k)
    alpha = 0.002
    for i in range(256):
        for j in range(256):
            Scmp[i,j] = Scm[i,j] + alpha*Awa[i,j]
    
    Aw = np.dot(Uc, Scmp)
    Aw = np.dot(Aw, Vc)
    
    AwF = np.zeros(k, dtype=object)
    for i in range(256):
        for j in range(256):
            AwF[i,j] = int(np.floor(Aw[i,j]))
    
    return AwF

def generate_chaos_matrix(size):
    M = np.zeros((size, size), dtype=object)
    codem = random.uniform(0, 1)
    var = codem
    d = random.uniform(0, 1)
    
    for j in range(size):
        for i in range(size):
            x = var
            y = tent(x, d)
            var = y
            M[i][j] = decimalToBinary((math.ceil(var*(10**9)))%4)
    return M, codem

def embed_watermark(watermark_gray, cover_gray_512):
    # Create 256x256 version of cover image for first stage
    cover_gray_256 = rescale_image(cover_gray_512, (256, 256))
    
    # Process watermark using SVD
    AwF = process_watermark(watermark_gray, cover_gray_256)
    
    # Create expanded bit matrix (2x2 blocks)
    exp2bit = np.zeros([512, 512], dtype=object)
    l = 0
    for i in range(256):
        k = 0
        for j in range(256):
            bits = format(int(AwF[i,j] % 256), '08b')
            exp2bit[l,k] = bits[0:2]
            exp2bit[l,k+1] = bits[2:4]
            exp2bit[l+1,k] = bits[4:6]
            exp2bit[l+1,k+1] = bits[6:8]
            k += 2
        l += 2
    
    # Generate chaos matrix
    M, codem = generate_chaos_matrix(512)
    
    # Apply CNOT operation
    EX = np.copy(exp2bit)
    for i in range(512):
        for j in range(512):
            for k in range(2):
                if M[i,j][k] == '1':
                    l = list(EX[i,j])
                    if len(l) >= 2:
                        l[k] = NOT(l[k])
                        EX[i,j] = "".join(l)
    
    # Convert original 512x512 cover image to binary
    imCbinary = np.zeros([512, 512], dtype=object)
    for i in range(512):
        for j in range(512):
            imCbinary[i,j] = format(cover_gray_512[i,j], '08b')
    
    # Embedding process
    key = np.zeros((512, 512), dtype=object)
    for i in range(512):
        for j in range(512):
            if len(EX[i,j]) >= 2 and len(imCbinary[i,j]) >= 8:
                key[i,j] = CNOT(imCbinary[i,j][4], EX[i,j][1])
                imCbinary[i,j] = imCbinary[i,j][:7] + CNOT(imCbinary[i,j][3], EX[i,j][0])
    
    # Convert back to image
    IMG = np.zeros((512, 512))
    for i in range(512):
        for j in range(512):
            try:
                IMG[i,j] = int(imCbinary[i,j], 2)
            except ValueError:
                IMG[i,j] = 0
    
    return IMG, key
    
def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def extract_watermark(cover_gray_512, key, imCbinary, codem):
    a = (512, 512)
    KY = np.copy(key)
    imp = np.zeros(a, dtype=object)

    for i in range(512):
        for j in range(512):
            imp[i, j] = CNOT(imCbinary[i, j][3], imCbinary[i, j][7]) + CNOT(imCbinary[i, j][4], KY[i, j])

    # Preparing chaos matrix for extraction
    ab = (512, 512)
    CME = np.zeros(ab, dtype=object)
    varx = codem

    for j in range(512):
        for i in range(512):
            x = varx
            y = tent(x, 1)  # Assuming d=1 for tent function
            varx = y
            CME[i][j] = decimalToBinary((math.ceil(varx * (10 ** 9))) % 4)

    IMB = np.copy(imp)

    for i in range(len(CME)):
        for j in range(len(CME)):
            for k in range(2):
                if CME[i, j][k] == '1':
                    l = list(IMB[i, j])
                    l[k] = NOT(l[k])
                    IMB[i, j] = "".join(l)

    # IMB is the scrambled watermarked image
    ab = (256, 256)
    imb = np.zeros(ab, dtype=object)
    l = 0

    for i in range(256):
        k = 0
        for j in range(256):
            imb[i, j] = IMB[l, k] + IMB[l, k + 1] + IMB[l + 1, k] + IMB[l + 1, k + 1]
            k += 2
        l += 2

    lx = (256, 256)
    IMGx = np.zeros(lx)

    for i in range(256):
        for j in range(256):
            IMGx[i, j] = int(imb[i, j], 2)

    return IMGx

def main():
    st.title("Digital Image Watermarking")
    st.write("Select a cover image and a watermark to embed.")
    
    # Dropdown for selecting cover images
    cover_image_options = [
        "australia.png",
        "Boat.png",
        "Butterfly.jpg",
        "casa.png",
        "fachada.png",
        "owl.png"
    ]
    
    watermark_image_options = [
        "Watermark_1.jpg",
        "watermark_2.png",
        "watermark_3.png"
    ]
    
    selected_cover_image = st.selectbox("Choose a cover image", cover_image_options)
    selected_watermark_image = st.selectbox("Choose a watermark image", watermark_image_options)
    
    if selected_cover_image and selected_watermark_image:
        try:
            # Load images from the specified folders
            cover_img_path = f"IMAGES/{selected_cover_image}"
            watermark_img_path = f"watermarks/{selected_watermark_image}"
            
            cover_img = cv2.imread(cover_img_path)
            watermark_img = cv2.imread(watermark_img_path)
            
            # Convert to grayscale
            cover_gray = convert_to_grayscale(cover_img)
            watermark_gray = convert_to_grayscale(watermark_img)
            
            # Resize watermark to 256x256 and cover to 512x512
            watermark_gray = rescale_image(watermark_gray, (256, 256))
            cover_gray = rescale_image(cover_gray, (512, 512))
            
            # Display original images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cover Image (512x512)")
                st.image(cover_gray, use_container_width=True)
            
            with col2:
                st.subheader("Watermark (256x256)")
                st.image(watermark_gray, use_container_width=True)
            
            if st.button("Embed Watermark"):
                with st.spinner("Processing..."):
                    watermarked_img, key = embed_watermark(watermark_gray, cover_gray)
                    
                    if watermarked_img is not None:
                        psnr = calculate_psnr(cover_gray, watermarked_img)
                        
                        st.subheader("Watermarked Image")
                        st.image(watermarked_img.astype(np.uint8), use_container_width=True)
                        
                        if psnr is not None:
                            st.write(f"PSNR: {psnr:.4f} dB")

            if st.button("Extract Watermark"):
                with st.spinner("Extracting..."):
                    imCbinary = np.zeros([512, 512], dtype=object)

                    for i in range(512):
                        for j in range(512):
                            imCbinary[i, j] = format(cover_gray[i, j], '08b')

                    extracted_watermark = extract_watermark(cover_gray, key, imCbinary, codem)

                    st.subheader("Extracted Watermark")
                    st.image(extracted_watermark.astype(np.uint8), use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure both images are valid and try again.")

if __name__ == "__main__":
    main()
