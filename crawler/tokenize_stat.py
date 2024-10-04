import os
import nltk
from nltk.tokenize import word_tokenize
from loguru import logger
import tiktoken

def tokenize_nltk(content):
    return word_tokenize(content)

def tokenize_tiktoken(content):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(content)

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        nltk_tokens = tokenize_nltk(content)
        tiktoken_tokens = tokenize_tiktoken(content)
        return len(nltk_tokens), len(tiktoken_tokens)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0, 0

def process_documents(root_directory, lang_code):
    total_nltk_tokens = 0
    total_tiktoken_tokens = 0
    file_count = 0

    lang_directory = os.path.join(root_directory, lang_code)
    
    if not os.path.exists(lang_directory):
        logger.error(f"Directory for language '{lang_code}' not found.")
        return

    for file in os.listdir(lang_directory)[:5]:  # Process only the first 5 articles
        if file.endswith('.txt'):
            file_path = os.path.join(lang_directory, file)
            nltk_count, tiktoken_count = process_file(file_path)
            total_nltk_tokens += nltk_count
            total_tiktoken_tokens += tiktoken_count
            file_count += 1
            logger.info(f"File: {file_path}, NLTK Tokens: {nltk_count}, Tiktoken Tokens: {tiktoken_count}")

        if file_count == 5:  # Break after processing 5 files
            break

    if file_count > 0:
        average_nltk_tokens = total_nltk_tokens / file_count
        average_tiktoken_tokens = total_tiktoken_tokens / file_count
        logger.info(f"Total files processed for {lang_code}: {file_count}")
        logger.info(f"Total NLTK tokens for {lang_code}: {total_nltk_tokens}")
        logger.info(f"Total Tiktoken tokens for {lang_code}: {total_tiktoken_tokens}")
        logger.info(f"Average NLTK tokens per file for {lang_code}: {average_nltk_tokens:.2f}")
        logger.info(f"Average Tiktoken tokens per file for {lang_code}: {average_tiktoken_tokens:.2f}")
    else:
        logger.warning(f"No files were processed for language '{lang_code}'.")

if __name__ == "__main__":
    nltk.download('punkt_tab', quiet=True)
    root_directory = "wikipedia_content"
    lang_code = "zh"  # Change this to the desired language code
    process_documents(root_directory, lang_code)
