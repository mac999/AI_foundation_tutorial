# 정규식 사용해 CSV 파일에서 단어를 추출
# https://docs.python.org/3/library/re.html
import os
import re
import csv

def extract_words_from_csv(folder_path, regex_pattern):
    """
    Reads all CSV files in the specified folder and extracts words matching the regex pattern.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        regex_pattern (str): Regular expression pattern to match words.

    Returns:
        list: A list of matched words from all CSV files.
    """
    matched_words = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            
            # Open and read the CSV file
            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    for cell in row:
                        ret = re.search(regex_pattern, cell)
                        if ret == None:
                            continue

                        match = {
                            'matches': ret,
                            'file': file_name,
                            'row': row,
                            'cell': cell
                        }
                        matched_words.append(match)
    
    return matched_words

# Example usage
if __name__ == "__main__":
    # Define the folder path and regex pattern
    folder_path = "./data"
    regex_pattern = r'\bGPU\b'  # Matches the word 'GPU'

    # Extract words
    extracted_words = extract_words_from_csv(folder_path, regex_pattern)

    # Print the results
    print("Extracted Words:", extracted_words)