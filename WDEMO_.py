import streamlit as st
import numpy as np
import cv2
import random
import math
from scipy import linalg
from io import BytesIO

# Configuration
COVER_IMAGE_SIZE = (512, 512)
WATERMARK_SIZE = (256, 256)
ALPHA = 0.002  # Watermark strength factor

class ImageProcessor:
    """Handles basic image processing operations"""
    
    @staticmethod
    def convert_to_grayscale(image):
        """Convert BGR image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def rescale_image(img, size):
        """
        Rescale image to specified size using Lanczos interpolation
        Args:
            img: Input image
            size: Target size tuple (width, height)
        """
        return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def load_image_from_upload(uploaded_file):
        """
        Convert uploaded file to OpenCV image
        Args:
            uploaded_file: Streamlit uploaded file object
        """
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

class QuantumOperations:
    """Implements quantum gate operations"""
    
    @staticmethod
    def NOT(x):
        """Quantum NOT gate implementation"""
        return "1" if x == "0" else "0"
    
    @staticmethod
    def CNOT(x, y):
        """
        Quantum CNOT gate implementation
        Args:
            x: Control bit
            y: Target bit
        """
        return QuantumOperations.NOT(y) if x == "1" else y

class ChaosGenerator:
    """Handles chaos-based encryption operations"""
    
    @staticmethod
    def tent(x, d):
        """
        Tent map function for chaos generation
        Args:
            x: Input value
            d: Control parameter
        """
        if 0 <= x < d:
            return x/d
        elif d <= x < 1:
            return (1-x)/(1-d)
        return 0
    
    @staticmethod
    def decimal_to_binary(n):
        """Convert decimal to 2-bit binary string"""
        return "{0:02b}".format(int(n))
    
    @staticmethod
    def generate_chaos_matrix(size):
        """
        Generate chaos matrix for encryption
        Args:
            size: Size of the matrix to generate
        Returns:
            Tuple of (chaos matrix, initial code)
        """
        M = np.zeros((size, size), dtype=object)
        codem = random.uniform(0, 1)
        var = codem
        d = random.uniform(0, 1)
        
        for j in range(size):
            for i in range(size):
                x = var
                y = ChaosGenerator.tent(x, d)
                var = y
                M[i][j] = ChaosGenerator.decimal_to_binary((math.ceil(var*(10**9)))%4)
        return M, codem

class WatermarkProcessor:
    """Main watermark processing class"""
    
    @staticmethod
    def process_watermark(watermark_gray, cover_gray_256):
        """
        Process watermark using SVD
        Args:
            watermark_gray: Grayscale watermark image
            cover_gray_256: 256x256 grayscale cover image
        """
        # SVD Process
        A = watermark_gray.astype(float)
        I = cover_gray_256.astype(float)
        
        Uw, Sw, Vw = linalg.svd(A, full_matrices=True)
        Uc, Sc, Vc = linalg.svd(I, full_matrices=True)
        
        k = WATERMARK_SIZE
        Swm = np.zeros(k)
        Scm = np.zeros(k)
        
        for i in range(min(256, len(Sw))):
            Swm[i,i] = Sw[i]
            Scm[i,i] = Sc[i]
        
        Awa = np.dot(Uw, Swm)
        
        Scmp = np.zeros(k)
        for i in range(256):
            for j in range(256):
                Scmp[i,j] = Scm[i,j] + ALPHA*Awa[i,j]
        
        Aw = np.dot(Uc, Scmp)
        Aw = np.dot(Aw, Vc)
        
        AwF = np.zeros(k, dtype=object)
        for i in range(256):
            for j in range(256):
                AwF[i,j] = int(np.floor(Aw[i,j]))
        
        return AwF
    
    @staticmethod
    def embed_watermark(watermark_gray, cover_gray_512):
        """
        Embed watermark into cover image
        Args:
            watermark_gray: Grayscale watermark image
            cover_gray_512: 512x512 grayscale cover image
        """
        # Create 256x256 version of cover image
        cover_gray_256 = ImageProcessor.rescale_image(cover_gray_512, WATERMARK_SIZE)
        
        # Process watermark using SVD
        AwF = WatermarkProcessor.process_watermark(watermark_gray, cover_gray_256)
        
        # Create expanded bit matrix (2x2 blocks)
        exp2bit = np.zeros(COVER_IMAGE_SIZE, dtype=object)
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
        
        # Generate and apply chaos matrix
        M, codem = ChaosGenerator.generate_chaos_matrix(512)
        EX = WatermarkProcessor.apply_quantum_operations(exp2bit, M)
        
        # Convert cover image to binary and embed watermark
        imCbinary = WatermarkProcessor.convert_to_binary(cover_gray_512)
        IMG, key = WatermarkProcessor.perform_embedding(EX, imCbinary)
        
        return IMG, key, codem, imCbinary
    
    @staticmethod
    def apply_quantum_operations(exp2bit, chaos_matrix):
        """Apply quantum operations using chaos matrix"""
        EX = np.copy(exp2bit)
        for i in range(512):
            for j in range(512):
                for k in range(2):
                    if chaos_matrix[i,j][k] == '1':
                        l = list(EX[i,j])
                        if len(l) >= 2:
                            l[k] = QuantumOperations.NOT(l[k])
                            EX[i,j] = "".join(l)
        return EX
    
    @staticmethod
    def convert_to_binary(image):
        """Convert image to binary representation"""
        binary = np.zeros(COVER_IMAGE_SIZE, dtype=object)
        for i in range(512):
            for j in range(512):
                binary[i,j] = format(image[i,j], '08b')
        return binary
    
    @staticmethod
    def perform_embedding(EX, imCbinary):
        """Perform the actual watermark embedding"""
        key = np.zeros(COVER_IMAGE_SIZE, dtype=object)
        IMG = np.zeros(COVER_IMAGE_SIZE)
        
        for i in range(512):
            for j in range(512):
                if len(EX[i,j]) >= 2 and len(imCbinary[i,j]) >= 8:
                    key[i,j] = QuantumOperations.CNOT(imCbinary[i,j][4], EX[i,j][1])
                    imCbinary[i,j] = imCbinary[i,j][:7] + QuantumOperations.CNOT(imCbinary[i,j][3], EX[i,j][0])
                try:
                    IMG[i,j] = int(imCbinary[i,j], 2)
                except ValueError:
                    IMG[i,j] = 0
                    
        return IMG, key

class Metrics:
    """Handles quality metrics calculations"""
    
    @staticmethod
    def calculate_psnr(original, watermarked):
        """
        Calculate Peak Signal-to-Noise Ratio
        Args:
            original: Original image
            watermarked: Watermarked image
        """
        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))

class StreamlitInterface:
    """Handles the Streamlit user interface"""
    
    def __init__(self):
        self.cover_image_options = [
            "australia.png", "Boat.png", "Butterfly.jpg",
            "casa.png", "fachada.png", "owl.png"
        ]
        self.watermark_image_options = [
            "Watermark_1.jpg", "watermark_2.png", "watermark_3.png"
        ]
    
    def render(self):
        """Render the Streamlit interface"""
        st.title("Two-Step Hybrid Quantum Watermarking System")
        st.write("Select or upload a cover image and a watermark to embed.")
        
        # Create two-column layout
        col1, col2 = st.columns(2)
        
        # Handle cover image selection
        cover_img = self._handle_cover_image(col1)
        
        # Handle watermark selection
        watermark_img = self._handle_watermark(col2)
        
        # Process images if both are available
        if cover_img is not None and watermark_img is not None:
            self._process_images(cover_img, watermark_img)
    
    def _handle_cover_image(self, column):
        """Handle cover image selection/upload"""
        with column:
            st.subheader("Cover Image")
            cover_source = st.radio("Choose cover image source:", 
                                  ["Select from library", "Upload custom image"])
            
            if cover_source == "Select from library":
                selected_cover = st.selectbox("Choose a cover image", 
                                            self.cover_image_options)
                return cv2.imread(f"IMAGES/{selected_cover}") if selected_cover else None
            else:
                uploaded_cover = st.file_uploader("Upload cover image (512x512)", 
                                                type=['png', 'jpg', 'jpeg'])
                return ImageProcessor.load_image_from_upload(uploaded_cover) if uploaded_cover else None
    
    def _handle_watermark(self, column):
        """Handle watermark selection/upload"""
        with column:
            st.subheader("Watermark")
            watermark_source = st.radio("Choose watermark source:", 
                                      ["Select from library", "Upload custom image"])
            
            if watermark_source == "Select from library":
                selected_watermark = st.selectbox("Choose a watermark image", 
                                                self.watermark_image_options)
                return cv2.imread(f"watermarks/{selected_watermark}") if selected_watermark else None
            else:
                uploaded_watermark = st.file_uploader("Upload watermark image (256x256)", 
                                                    type=['png', 'jpg', 'jpeg'])
                return ImageProcessor.load_image_from_upload(uploaded_watermark) if uploaded_watermark else None
    
    def _process_images(self, cover_img, watermark_img):
        """Process and display the images"""
        try:
            # Convert and resize images
            cover_gray = ImageProcessor.convert_to_grayscale(cover_img)
            watermark_gray = ImageProcessor.convert_to_grayscale(watermark_img)
            
            watermark_gray = ImageProcessor.rescale_image(watermark_gray, WATERMARK_SIZE)
            cover_gray = ImageProcessor.rescale_image(cover_gray, COVER_IMAGE_SIZE)
            
            # Display original images
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Cover Image (512x512)")
                st.image(cover_gray, use_container_width=True)
            
            with col4:
                st.subheader("Watermark (256x256)")
                st.image(watermark_gray, use_container_width=True)
            
            # Process watermark embedding
            if st.button("Embed Watermark"):
                with st.spinner("Processing..."):
                    watermarked_img, key, codem, imCbinary = WatermarkProcessor.embed_watermark(
                        watermark_gray, cover_gray
                    )
                    
                    if watermarked_img is not None:
                        psnr = Metrics.calculate_psnr(cover_gray, watermarked_img)
                        
                        st.subheader("Watermarked Image")
                        st.image(watermarked_img.astype(np.uint8), use_container_width=True)
                        
                        if psnr is not None:
                            st.write(f"PSNR: {psnr:.4f} dB")


        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure both images are valid and try again.")

def main():
    """Main application entry point"""
    interface = StreamlitInterface()
    interface.render()

if __name__ == "__main__":
    main()
