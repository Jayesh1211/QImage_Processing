import streamlit as st
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from scipy import linalg
import random
import math

def convert_to_grayscale(img_array):
    """Convert image array to grayscale, handling both RGB and grayscale inputs"""
    # First resize the image to 256x256
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    # Then convert to grayscale if needed
    if len(img_array.shape) == 3:  # If image is RGB
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:  # If image is already grayscale
        return img_array

def tent(x):
    d = random.uniform(0, 1)
    if x >= 0 and x < d:
        y = x/d
    elif x >= d and x < 1:
        y = (1-x)/(1-d)
    return y

def decimalToBinary(n):
    return "{0:02b}".format(int(n))

def NOT(x):
    if x == "0":
        x = "1"
    else:
        x = "0"
    return x

def CNOT(x, y):
    if x == "1":
        y = NOT(y)
    elif x == "0":
        y = y
    return y

def modadd(x, y):
    if x == '0' and y == '0':
        return '0'
    if x == '0' and y == '1':
        return '1'
    if x == '1' and y == '0':
        return '1'
    if x == '1' and y == '1':
        return '0'

def main():
    st.title("Digital Image Watermarking Application")
    
    # File uploaders
    watermark_file = st.file_uploader("Upload Watermark Image", type=['png', 'jpg', 'jpeg'])
    cover_file = st.file_uploader("Upload Cover Image", type=['png', 'jpg', 'jpeg'])
    
    if watermark_file and cover_file:
        try:
            # Process watermark image
            watermark_img = Image.open(watermark_file)
            watermark_array = np.array(watermark_img)
            graywm = convert_to_grayscale(watermark_array)
            
            # Process cover image
            cover_img = Image.open(cover_file)
            cover_array = np.array(cover_img)
            grayc = convert_to_grayscale(cover_array)
            
            # Normalize pixel values to 0-255 range
            graywm = ((graywm - graywm.min()) * (255.0 / (graywm.max() - graywm.min()))).astype(np.uint8)
            grayc = ((grayc - grayc.min()) * (255.0 / (grayc.max() - grayc.min()))).astype(np.uint8)
            
            # Display original images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Watermark")
                st.image(graywm, use_container_width=True)
            with col2:
                st.subheader("Original Cover Image")
                st.image(grayc, use_container_width=True)
            
            if st.button("Apply Watermark"):
                with st.spinner("Processing watermark..."):
                    # SVD Process
                    A = np.copy(graywm)
                    Uw, Sw, Vw = linalg.svd(A, full_matrices=True)
                    
                    I = np.copy(grayc)
                    Uc, Sc, Vc = linalg.svd(I, full_matrices=True)
                    
                    k = (256, 256)
                    Swm = np.zeros(k)
                    Scm = np.zeros(k)
                    for i in range(256):
                        Swm[i,i] = Sw[i]
                        Scm[i,i] = Sc[i]
                    
                    Awa = np.zeros(k)
                    Awa = np.dot(Uw, Swm)
                    Scmp = np.zeros(k)
                    alpha = 0.002
                    for i in range(256):
                        for j in range(256):
                            Scmp[i,j] = Scm[i,j] + alpha*Awa[i,j]
                    
                    Aw = np.zeros(k)
                    Aw = np.dot(Uc, Scmp)
                    Aw = np.dot(Aw, Vc)
                    
                    AwF = np.zeros(k, dtype=object)
                    Awf = np.zeros(k)
                    for i in range(256):
                        for j in range(256):
                            AwF[i,j] = int(np.floor(Aw[i,j]))
                            Awf[i,j] = Aw[i,j] - AwF[i,j]
                    
                    # Binary conversion and expansion
                    exp2bit = np.zeros([512,512], dtype=object)
                    l = 0
                    for i in range(256):
                        k = 0
                        for j in range(256):
                            exp2bit[l,k] = format(int(abs(AwF[i,j])),'08b')[0:2]
                            exp2bit[l,k+1] = format(int(abs(AwF[i,j])),'08b')[2:4]
                            exp2bit[l+1,k] = format(int(abs(AwF[i,j])),'08b')[4:6]
                            exp2bit[l+1,k+1] = format(int(abs(AwF[i,j])),'08b')[6:8]
                            k += 2
                        l += 2
                    
                    # Rest of the processing code remains the same...
                    n = 9
                    size = 2**n
                    M = np.zeros((size,size), dtype=object)
                    codem = random.uniform(0,1)
                    var = codem
                    for j in range(size):
                        for i in range(size):
                            x = var
                            y = tent(x)
                            var = y
                            M[i][j] = decimalToBinary((math.ceil(var*(10**9)))%4)
                    
                    # Scrambling
                    EX = np.copy(exp2bit)
                    for i in range(len(M)):
                        for j in range(len(M)):
                            for k in range(2):
                                if M[i,j][k] == '1':
                                    l = list(EX[i,j])
                                    l[k] = NOT(l[k])
                                    EX[i,j] = "".join(l)
                    
                    # Calculate pixel difference matrix
                    CM = np.zeros((256,256))
                    l = 0
                    for i in range(256):
                        k = 0
                        for j in range(256):
                            x = []
                            x.append(grayc[l,k])
                            x.append(grayc[l,k+1])
                            x.append(grayc[l+1,k])
                            x.append(grayc[l+1,k+1])
                            CM[i,j] = max(x) - min(x)
                            k += 2
                        l += 2
                    
                    # Determine smooth blocks
                    sm = np.zeros((256,256), dtype=object)
                    for i in range(256):
                        for j in range(256):
                            if CM[i,j] <= 15:
                                sm[i,j] = '1'
                            else:
                                sm[i,j] = '0'
                    
                    # Convert cover image to binary
                    imCbinary = np.zeros([512,512], dtype=object)
                    for i in range(512):
                        for j in range(512):
                            imCbinary[i,j] = format(grayc[i//2,j//2], '08b')
                    
                    # Embedding process
                    key = np.zeros((512,512), dtype=object)
                    for i in range(512):
                        for j in range(512):
                            key[i,j] = CNOT(imCbinary[i,j][4], EX[i,j][1])
                            imCbinary[i,j] = imCbinary[i,j][:7] + CNOT(imCbinary[i,j][3], EX[i,j][0])
                    
                    # Convert to final image
                    IMG = np.zeros((512,512))
                    for i in range(512):
                        for j in range(512):
                            IMG[i,j] = int(imCbinary[i,j], 2)
                    
                    # Calculate PSNR
                    MSEi = 0
                    for i in range(512):
                        for j in range(512):
                            MSEi += (grayc[i//2,j//2]-IMG[i,j])**2
                    gl = 512*512
                    MSE = MSEi/gl
                    MAX = 255
                    PSNR = 10*math.log(MAX**2/MSE, 10)
                    
                    # Display results
                    st.subheader("Watermarked Image")
                    st.image(IMG.astype(np.uint8), use_container_width=True)
                    st.success(f"PSNR Value: {PSNR:.2f} dB")
                    
                    # Count smooth blocks
                    smooth_blocks = np.sum(sm == '1')
                    st.info(f"Number of smooth blocks: {smooth_blocks}")
                    st.info(f"Number of edge blocks: {256*256 - smooth_blocks}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure your images are in the correct format and try again.")

if __name__ == "__main__":
    main()
