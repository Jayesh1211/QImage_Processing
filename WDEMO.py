import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random
import math
from scipy import linalg
import io

def convert_to_grayscale(image):
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        st.error("Error converting image to grayscale. Please check the input image.")
        return None

def rescale_image(img, size=(256, 256)):
    try:
        return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    except cv2.error:
        st.error("Error resizing image. Please check the input image.")
        return None

def tent(x, d):
    try:
        if 0 <= x < d:
            return x/d
        elif d <= x < 1:
            return (1-x)/(1-d)
        return 0
    except ZeroDivisionError:
        return 0

def decimalToBinary(n):
    try:
        return "{0:02b}".format(int(n))
    except ValueError:
        return "00"

def NOT(x):
    return "1" if x == "0" else "0"

def CNOT(x, y):
    return NOT(y) if x == "1" else y

def generate_chaos_matrix(size, seed):
    try:
        M = np.zeros((size, size), dtype=object)
        var = seed
        d = random.uniform(0, 1)
        
        for j in range(size):
            for i in range(size):
                x = var
                y = tent(x, d)
                var = y
                M[i][j] = decimalToBinary((math.ceil(var*(10**9)))%4)
        return M
    except Exception as e:
        st.error(f"Error generating chaos matrix: {str(e)}")
        return None

def process_watermark(watermark_img, cover_img):
    try:
        # SVD Process
        A = np.copy(watermark_img)
        I = np.copy(cover_img)
        
        Uw, Sw, Vw = linalg.svd(A, full_matrices=True)
        Uc, Sc, Vc = linalg.svd(I, full_matrices=True)
        
        k = (256, 256)
        Swm = np.zeros(k)
        Scm = np.zeros(k)
        for i in range(256):
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
        
        Awf = np.zeros(k)
        AwF = np.zeros(k, dtype=object)
        for i in range(256):
            for j in range(256):
                AwF[i,j] = int(np.floor(Aw[i,j]))
                Awf[i,j] = Aw[i,j] - AwF[i,j]
        
        return AwF
    except Exception as e:
        st.error(f"Error processing watermark: {str(e)}")
        return None

def embed_watermark(watermark_gray, cover_gray):
    try:
        # Process watermark
        AwF = process_watermark(watermark_gray, cover_gray)
        if AwF is None:
            return None, None
        
        # Generate expanded bit matrix
        exp2bit = np.zeros([512,512], dtype=object)
        l = 0
        for i in range(256):
            k = 0
            for j in range(256):
                exp2bit[l,k] = format(((AwF[i,j])),'08b')[0:2]
                exp2bit[l,k+1] = format(((AwF[i,j])),'08b')[2:4]
                exp2bit[l+1,k] = format(((AwF[i,j])),'08b')[4:6]
                exp2bit[l+1,k+1] = format(((AwF[i,j])),'08b')[6:8]
                k+=2
            l+=2
        
        # Generate chaos matrix
        M = generate_chaos_matrix(512, random.uniform(0,1))
        if M is None:
            return None, None
        
        # Apply CNOT operation
        EX = np.copy(exp2bit)
        for i in range(len(M)):
            for j in range(len(M)):
                for k in range(2):
                    if M[i,j][k]=='1':
                        l = list(EX[i,j])
                        l[k] = NOT(l[k])
                        EX[i,j] = "".join(l)
        
        # Convert cover image to binary
        imCbinary = np.zeros([512,512], dtype=object)
        for i in range(512):
            for j in range(512):
                imCbinary[i,j] = format(cover_gray[i,j], '08b')
        
        # Embedding process
        key = np.zeros((512,512), dtype=object)
        for i in range(512):
            for j in range(512):
                key[i,j] = CNOT(imCbinary[i,j][4], EX[i,j][1])
                imCbinary[i,j] = imCbinary[i,j][:7] + CNOT(imCbinary[i,j][3], EX[i,j][0])
        
        # Convert back to image
        IMG = np.zeros((512,512))
        for i in range(512):
            for j in range(512):
                IMG[i,j] = int(imCbinary[i,j], 2)
        
        return IMG, key
    except Exception as e:
        st.error(f"Error embedding watermark: {str(e)}")
        return None, None

def calculate_psnr(original, watermarked):
    try:
        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    except Exception as e:
        st.error(f"Error calculating PSNR: {str(e)}")
        return None

def main():
    st.title("Digital Image Watermarking")
    st.write("Upload a cover image and a watermark to embed.")
    
    cover_file = st.file_uploader("Choose a cover image", type=['png', 'jpg', 'jpeg'])
    watermark_file = st.file_uploader("Choose a watermark image", type=['png', 'jpg', 'jpeg'])
    
    if cover_file and watermark_file:
        try:
            # Read and process cover image
            cover_bytes = np.asarray(bytearray(cover_file.read()), dtype=np.uint8)
            cover_img = cv2.imdecode(cover_bytes, cv2.IMREAD_COLOR)
            if cover_img is None:
                st.error("Error reading cover image. Please check the file.")
                return
                
            cover_gray = convert_to_grayscale(cover_img)
            if cover_gray is None:
                return
                
            cover_gray = rescale_image(cover_gray)
            if cover_gray is None:
                return
            
            # Read and process watermark image
            watermark_bytes = np.asarray(bytearray(watermark_file.read()), dtype=np.uint8)
            watermark_img = cv2.imdecode(watermark_bytes, cv2.IMREAD_COLOR)
            if watermark_img is None:
                st.error("Error reading watermark image. Please check the file.")
                return
                
            watermark_gray = convert_to_grayscale(watermark_img)
            if watermark_gray is None:
                return
                
            watermark_gray = rescale_image(watermark_gray)
            if watermark_gray is None:
                return
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cover Image")
                st.image(cover_gray, use_container_width=True)
            
            with col2:
                st.subheader("Watermark")
                st.image(watermark_gray, use_container_width=True)
            
            if st.button("Embed Watermark"):
                with st.spinner("Processing..."):
                    watermarked_img, key = embed_watermark(watermark_gray, cover_gray)
                    
                    if watermarked_img is not None:
                        psnr = calculate_psnr(cover_gray, watermarked_img)
                        
                        st.subheader("Watermarked Image")
                        st.image(watermarked_img.astype(np.uint8), use_container_width=True)
                        
                        if psnr is not None:
                            st.write(f"PSNR: {psnr:.2f} dB")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
