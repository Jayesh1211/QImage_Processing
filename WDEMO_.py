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
        """Rescale image to specified size using Lanczos interpolation"""
        return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def load_image_from_upload(uploaded_file):
        """Convert uploaded file to OpenCV image"""
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
        """Quantum CNOT gate implementation"""
        return QuantumOperations.NOT(y) if x == "1" else y

class ChaosGenerator:
    """Handles chaos-based encryption operations"""
    
    @staticmethod
    def tent(x, d):
        """Tent map function for chaos generation"""
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
        """Generate chaos matrix for encryption"""
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
        """Process watermark using SVD"""
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
        """Embed watermark into cover image"""
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
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))


class StreamlitInterface:
    """Handles the Streamlit user interface with enhanced styling"""
    
    def __init__(self):
        self.cover_image_options = [
            "australia.png", "Boat.png", "Butterfly.jpg",
            "casa.png", "fachada.png", "owl.png"
        ]
        self.watermark_image_options = [
            "Watermark_1.jpg", "watermark_2.png", "watermark_3.png"
        ]
        self.setup_page()
    
    def setup_page(self):
        """Configure page settings and styling"""
        st.set_page_config(
            page_title="Quantum Watermarking System",
            page_icon="🔒",
            layout="wide"
        )
        
        # Custom CSS
        st.markdown("""
            <style>
                .stApp {
                    background: linear-gradient(to bottom right, #f5f7fa, #e3eeff);
                }
                .main {
                    padding: 2rem;
                }
                .stButton>button {
                    width: 100%;
                    border-radius: 10px;
                    background-color: #4CAF50;
                    color: white;
                    padding: 0.75rem;
                    margin: 1rem 0;
                }
                .stButton>button:hover {
                    background-color: #45a049;
                }
                h1 {
                    color: #1E3D59;
                    text-align: center;
                    padding: 1.5rem 0;
                    border-bottom: 2px solid #1E3D59;
                    margin-bottom: 2rem;
                }
                h2 {
                    color: #1E3D59;
                    margin: 1rem 0;
                }
                .stRadio > label {
                    padding: 0.5rem;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                    margin: 0.5rem 0;
                }
                .status-box {
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                }
                .metrics-box {
                    background-color: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)
    
    def render(self):
        """Render the enhanced Streamlit interface"""
        # Header with icon and description
        st.markdown("""
            <h1>🔒 Two-Step Hybrid Quantum Watermarking System</h1>
        """, unsafe_allow_html=True)
        
        # Info box with system description
        st.info("""
            This system uses quantum computing principles and chaos theory to securely embed 
            watermarks into images. Select or upload your images below to begin.
        """)
        
        # Main content container
        with st.container():
            # Image selection columns
            col1, col2 = st.columns(2)
            
            # Handle cover image selection
            cover_img = self._handle_cover_image(col1)
            
            # Handle watermark selection
            watermark_img = self._handle_watermark(col2)
            
            # Process images if both are available
            if cover_img is not None and watermark_img is not None:
                self._process_images(cover_img, watermark_img)
    
    def _handle_cover_image(self, column):
        """Handle cover image selection/upload with enhanced UI"""
        with column:
            st.markdown("""
                <h2>📄 Cover Image</h2>
            """, unsafe_allow_html=True)
            
            # Create tabs for selection methods
            tab1, tab2 = st.tabs(["📚 Select from Library", "⬆️ Upload Custom"])
            
            with tab1:
                selected_cover = st.selectbox(
                    "Choose from available images:",
                    self.cover_image_options,
                    key="cover_lib"
                )
                return cv2.imread(f"IMAGES/{selected_cover}") if selected_cover else None
                
            with tab2:
                uploaded_cover = st.file_uploader(
                    "Upload your cover image (512x512)",
                    type=['png', 'jpg', 'jpeg'],
                    key="cover_upload"
                )
                if uploaded_cover:
                    st.success("✅ Cover image uploaded successfully!")
                    return ImageProcessor.load_image_from_upload(uploaded_cover)
                return None
    
    def _handle_watermark(self, column):
        """Handle watermark selection/upload with enhanced UI"""
        with column:
            st.markdown("""
                <h2>💧 Watermark</h2>
            """, unsafe_allow_html=True)
            
            # Create tabs for selection methods
            tab1, tab2 = st.tabs(["📚 Select from Library", "⬆️ Upload Custom"])
            
            with tab1:
                selected_watermark = st.selectbox(
                    "Choose from available watermarks:",
                    self.watermark_image_options,
                    key="watermark_lib"
                )
                return cv2.imread(f"watermarks/{selected_watermark}") if selected_watermark else None
                
            with tab2:
                uploaded_watermark = st.file_uploader(
                    "Upload your watermark image (256x256)",
                    type=['png', 'jpg', 'jpeg'],
                    key="watermark_upload"
                )
                if uploaded_watermark:
                    st.success("✅ Watermark uploaded successfully!")
                    return ImageProcessor.load_image_from_upload(uploaded_watermark)
                return None
    
    def _process_images(self, cover_img, watermark_img):
        """Process and display the images with enhanced visualization"""
        try:
            # Convert and resize images
            cover_gray = ImageProcessor.convert_to_grayscale(cover_img)
            watermark_gray = ImageProcessor.convert_to_grayscale(watermark_img)
            
            watermark_gray = ImageProcessor.rescale_image(watermark_gray, WATERMARK_SIZE)
            cover_gray = ImageProcessor.rescale_image(cover_gray, COVER_IMAGE_SIZE)
            
            # Create three columns for image display
            col1, col2 = st.columns(2)
            
            # Display original images with enhanced styling
            with col1:
                st.markdown("""
                    <h2>📄 Cover Image (512x512)</h2>
                """, unsafe_allow_html=True)
                st.image(cover_gray, use_column_width=True)
            
            with col2:
                st.markdown("""
                    <h2>💧 Watermark (256x256)</h2>
                """, unsafe_allow_html=True)
                st.image(watermark_gray, use_column_width=True)
            
            # Add processing button with spinner
            if st.button("🔒 Embed Watermark", key="process_btn"):
                with st.spinner("🔄 Processing... Please wait..."):
                    watermarked_img, key, codem, imCbinary = WatermarkProcessor.embed_watermark(
                        watermark_gray, cover_gray
                    )
                    
                    if watermarked_img is not None:
                        # Calculate PSNR
                        psnr = Metrics.calculate_psnr(cover_gray, watermarked_img)
                        
                        # Display results in an organized layout
                        st.markdown("""
                            <h2>🎯 Results</h2>
                        """, unsafe_allow_html=True)
                        
                        # Display watermarked image
                        st.image(watermarked_img.astype(np.uint8), 
                                caption="Watermarked Image",
                                use_column_width=True)
                        
                        # Display metrics in a styled box
                        with st.container():
                            st.markdown("""
                                <div class='metrics-box'>
                                    <h3>📊 Quality Metrics</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if psnr is not None:
                                quality_level = "Excellent" if psnr > 40 else "Good" if psnr > 30 else "Fair"
                                st.metric(
                                    label="Peak Signal-to-Noise Ratio (PSNR)",
                                    value=f"{psnr:.2f} dB",
                                    delta=quality_level
                                )
                        
                        # Add download button for watermarked image
                        st.download_button(
                            label="📥 Download Watermarked Image",
                            data=cv2.imencode('.png', watermarked_img.astype(np.uint8))[1].tobytes(),
                            file_name="watermarked_image.png",
                            mime="image/png"
                        )
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please ensure both images are valid and try again.")

# [Rest of the code remains the same]

def main():
    """Main application entry point"""
    interface = StreamlitInterface()
    interface.render()

if __name__ == "__main__":
    main()
