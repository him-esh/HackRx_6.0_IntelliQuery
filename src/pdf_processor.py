# src/pdf_processor.py

from pypdf import PdfReader
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file.
    Returns an empty string if an error occurs.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
    return text

def process_all_pdfs_in_directory(input_dir, output_dir):
    """
    Processes all PDF files in the input_dir, extracts text,
    and saves it to corresponding .txt files in the output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    pdf_files_found = False
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"): # Case-insensitive check
            pdf_files_found = True
            pdf_path = os.path.join(input_dir, filename)
            output_txt_filename = f"{os.path.splitext(filename)[0]}.txt"
            output_txt_path = os.path.join(output_dir, output_txt_filename)

            print(f"Processing PDF: {filename}...")
            extracted_text = extract_text_from_pdf(pdf_path)

            if extracted_text.strip(): # Check if text is not just whitespace
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"  --> Successfully extracted and saved to: {output_txt_filename}")
            else:
                print(f"  --> No significant text extracted from {filename}. Check PDF or extraction method.")

    if not pdf_files_found:
        print(f"No PDF files found in '{input_dir}'. Please ensure your PDFs are in this folder.")


# This block allows the script to be run directly for testing.
if __name__ == "__main__":
    # Define paths relative to the project root
    # Navigate up one level from src (..) to get to the project root, then down to data/raw_pdfs
    INPUT_PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_pdfs')
    OUTPUT_TEXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'extracted_texts')

    print(f"Looking for PDFs in: {INPUT_PDF_DIR}")
    print(f"Saving extracted text to: {OUTPUT_TEXT_DIR}")

    if not os.path.exists(INPUT_PDF_DIR):
        print(f"\nERROR: Input PDF directory '{INPUT_PDF_DIR}' does not exist.")
        print("Please create this folder and place your PDF files inside it.")
    else:
        process_all_pdfs_in_directory(INPUT_PDF_DIR, OUTPUT_TEXT_DIR)
        print("\nPDF text extraction process completed.")