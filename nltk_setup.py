import nltk
import os

def download_nltk_data():
    # Set up NLTK data directory
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required NLTK data
    required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True, download_dir=nltk_data_dir)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {e}")
            raise

if __name__ == '__main__':
    download_nltk_data() 