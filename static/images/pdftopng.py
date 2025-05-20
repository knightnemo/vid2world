#!/usr/bin/env python3
import os
from pdf2image import convert_from_path
import glob

def convert_pdf_to_png():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all PDF files in the current directory
    pdf_files = glob.glob(os.path.join(current_dir, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert.")
    
    # Convert each PDF file
    for pdf_file in pdf_files:
        try:
            # Get the base name without extension
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            output_file = os.path.join(current_dir, f"{base_name}.png")
            
            print(f"Converting {pdf_file} to {output_file}...")
            
            # Convert PDF to image
            # Using 300 DPI for good quality
            images = convert_from_path(pdf_file, dpi=300)
            
            # Save the first page as PNG
            if images:
                images[0].save(output_file, "PNG")
                print(f"Successfully converted {pdf_file} to {output_file}")
            else:
                print(f"No pages found in {pdf_file}")
                
        except Exception as e:
            print(f"Error converting {pdf_file}: {str(e)}")

if __name__ == "__main__":
    convert_pdf_to_png()
