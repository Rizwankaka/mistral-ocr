"""
Streamlit app for OCR using Mistral AI
"""
import os
import tempfile
import streamlit as st
from pathlib import Path

from mistral_ocr import MistralOCR
from utils import get_combined_markdown, pretty_print_ocr

# Set page configuration
st.set_page_config(
    page_title="Mistral OCR App",
    page_icon="📄",
    layout="wide"
)

def main():
    """Main function for the Streamlit app."""
    
    # Header
    st.title("📄 Mistral OCR App")
    st.write("""
    This app uses Mistral AI's OCR capabilities to extract text and images from PDF files and images.
    Upload your files below to get started.
    """)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("Enter your Mistral API key", type="password")
        st.markdown("""
        ### How to get a Mistral API key:
        1. Go to [Mistral AI's platform](https://console.mistral.ai/)
        2. Sign up or log in
        3. Navigate to the API section
        4. Create a new API key
        """)
        
        st.divider()
        st.markdown("### Options")
        include_images = st.checkbox("Include images in results", value=True)
        show_raw_json = st.checkbox("Show raw JSON response", value=False)

    # Main content
    uploaded_file = st.file_uploader("Upload a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None and api_key:
        with st.spinner("Processing file..."):
            try:
                # Initialize MistralOCR client
                ocr_client = MistralOCR(api_key=api_key)
                
                # Process the file based on its type
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension == '.pdf':
                    ocr_response = ocr_client.process_pdf(
                        file_content=uploaded_file, 
                        file_name=uploaded_file.name,
                        include_images=include_images
                    )
                else:  # Image file
                    ocr_response = ocr_client.process_image(
                        file_content=uploaded_file,
                        file_name=uploaded_file.name
                    )
                
                # Display the OCR results
                st.success("File processed successfully!")
                
                # Display raw JSON if requested
                if show_raw_json:
                    with st.expander("Raw JSON Response"):
                        st.code(pretty_print_ocr(ocr_response), language="json")
                
                # Display the combined markdown with text and images
                st.subheader("OCR Results")
                st.markdown(get_combined_markdown(ocr_response), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    elif uploaded_file is not None and not api_key:
        st.warning("Please enter your Mistral API key in the sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Mistral AI | Created with Streamlit")

if __name__ == "__main__":
    main()