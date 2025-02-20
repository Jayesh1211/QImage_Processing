# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from scipy import linalg
import random
import math

# Function to rescale images
def rescale_image(input_image, size=(256, 256)):
    img_rescaled = input_image.resize(size, Image.Resampling.LANCZOS)
    return img_rescaled

# Function to convert image to grayscale
def convert_to_grayscale(image):
    # Convert to RGB if the image is in a single channel
    if image.mode == 'L':
        image = image.convert('RGB')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# Function to display images
def display_image(image, title):
    st.image(image, caption=title, use_column_width=True)

# Function to perform SVD
def perform_svd(image):
    A = np.copy(image)
    Uw, Sw, Vw = linalg.svd(A, full_matrices=True)
    return Uw, Sw, Vw

# Function to generate chaos matrix
def generate_chaos_matrix(n):
    k = 2 ** n
    M = np.zeros((k, k), dtype=object)
    codem = random.uniform(0, 1)
    var = codem
    for j in range(k):
        for i in range(k):
            x = var
            y = tent(x)
            var = y
            M[i][j] = decimalToBinary((math.ceil(var * (10 ** 9))) % 4)
    return M

# Tent function for chaos generation
def tent(x):
    d = random.uniform(0, 1)
    if x >= 0 and x < d:
        return x / d
    elif x >= d and x < 1:
        return (1 - x) / (1 - d)

# Function to convert decimal to binary
def decimalToBinary(n):
    return "{0:02b}".format(int(n))

# Function to apply CNOT operation
def CNOT(x, y):
    if x == "1":
        return NOT(y)
    return y

# Function to apply NOT operation
def NOT(x):
    return "1" if x == "0" else "0"

# Main Streamlit app
def main():
    st.title("Watermarking Application")
    
    # Sidebar for file uploads
    st.sidebar.header("Upload Images")
    watermark_file = st.sidebar.file_uploader("Upload Watermark Image", type=["png", "jpg", "jpeg"])
    cover_file = st.sidebar.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
    
    if watermark_file and cover_file:
        # Load and process watermark image
        WM = Image.open(watermark_file)
        graywm = convert_to_grayscale(WM)
        display_image(graywm, "Grayscale Watermark")
        
        # Load and process cover image
        CP = Image.open(cover_file)
        graycp = convert_to_grayscale(CP)
        display_image(graycp, "Grayscale Cover Image")
        
        # Rescale images
        rescaled_WM = rescale_image(WM)
        rescaled_CP = rescale_image(CP)
        
        # Perform SVD on watermark and cover images
        Uw, Sw, Vw = perform_svd(graywm)
        Uc, Sc, Vc = perform_svd(graycp)

        # Create matrices for watermark and cover
        k = (256, 256)
        Swm = np.zeros(k)
        Scm = np.zeros(k)
        for i in range(256):
            Swm[i, i] = Sw[i]
            Scm[i, i] = Sc[i]

        # Generate the watermarked image
        alpha = 0.002
        Awa = np.dot(Uw, Swm)
        Scmp = Scm + alpha * Awa
        Aw = np.dot(Uc, Scmp)
        AwF = np.floor(Aw).astype(int)

        # Convert pixel values into 8-bit string
        wmbinary = np.zeros([256, 256], dtype=object)
        for i in range(256):
            for j in range(256):
                wmbinary[i, j] = format(AwF[i, j], '08b')

        # Expand image according to protocol
        exp2bit = np.zeros([512, 512], dtype=object)
        l = 0
        for i in range(256):
            k = 0
            for j in range(256):
                exp2bit[l, k] = format(AwF[i, j], '08b')[0:2]
                exp2bit[l, k + 1] = format(AwF[i, j], '08b')[2:4]
                exp2bit[l + 1, k] = format(AwF[i, j], '08b')[4:6]
                exp2bit[l + 1, k + 1] = format(AwF[i, j], '08b')[6:8]
                k += 2
            l += 2

        # Generate chaos matrix
        n = 9
        M = generate_chaos_matrix(n)

        # Scrambling the image
        EX = np.copy(exp2bit)
        for i in range(len(M)):
            for j in range(len(M)):
                for k in range(2):
                    if M[i, j][k] == '1':
                        l = list(EX[i, j])
                        l[k] = NOT(l[k])
                        EX[i, j] = "".join(l)

        # Embedding process
        imCbinary = np.zeros([512, 512], dtype=object)
        for i in range(512):
            for j in range(512):
                imCbinary[i, j] = format(graycp[i, j], '08b')

        # Embedding starts
        key = np.zeros((512, 512), dtype=object)
        for i in range(512):
            for j in range(512):
                key[i, j] = CNOT(imCbinary[i, j][4], EX[i, j][1])
                imCbinary[i, j] = imCbinary[i, j][:7] + CNOT(imCbinary[i, j][3], EX[i, j][0])

        # Final image after embedding
        IMG = np.zeros((512, 512))
        for i in range(512):
            for j in range(512):
                IMG[i, j] = int(imCbinary[i, j], 2)

        # Display the final watermarked image
        st.write("Pixel value of embedded image:")
        st.image(IMG, caption="Embedded Image", use_column_width=True)

        # Calculate PSNR
        MSEi = np.mean((graycp - IMG) ** 2)
        MAX = 255
        PSNR = 10 * math.log10(MAX ** 2 / MSEi)
        st.write("The obtained PSNR value is:", PSNR)

if __name__ == "__main__":
    main()
