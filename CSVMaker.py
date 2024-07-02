import os
import csv
import argparse

def generate_csv(directory, output_csv):
    # Create and open the output CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        
        # Write the header
        writer.writerow(['image', 'prompt'])
        
        # Iterate through files in the specified directory
        for filename in os.listdir(directory):
            # Check for image files (you can add more formats if needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Get the corresponding text file name
                text_filename = os.path.splitext(filename)[0] + '.txt'
                text_filepath = os.path.join(directory, text_filename)
                
                # Check if the corresponding text file exists
                if os.path.isfile(text_filepath):
                    # Read the content of the text file
                    with open(text_filepath, 'r', encoding='utf-8') as text_file:
                        prompt = text_file.read().strip()
                    
                    # Write the image filename and prompt to the CSV
                    writer.writerow([filename, prompt])

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate a CSV file from images and text prompts.')
    parser.add_argument('-d', '--directory', required=True, help='Directory containing the images and text files.')
    parser.add_argument('-o', '--output', default='output.csv', help='Output CSV file name. Default is "output.csv".')
    
    args = parser.parse_args()
    
    # Generate the CSV file with the provided directory and output file name
    generate_csv(args.directory, args.output)