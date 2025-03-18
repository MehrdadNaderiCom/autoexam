import os
import random
import nltk
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with required NLTK data."""
        self.nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(self.nltk_data_dir):
            os.makedirs(self.nltk_data_dir, exist_ok=True)
        nltk.data.path.append(self.nltk_data_dir)

        # Download required NLTK data
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words('english'))

    def _ensure_nltk_data(self):
        """Ensure all required NLTK data is downloaded."""
        required_packages = {
            'punkt': 'tokenizers/punkt',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'stopwords': 'corpora/stopwords'
        }

        for package, path in required_packages.items():
            try:
                nltk.data.find(path)
                logger.info(f"Found existing {package} data")
            except LookupError:
                try:
                    logger.info(f"Downloading {package}...")
                    nltk.download(package, download_dir=self.nltk_data_dir, quiet=True)
                    logger.info(f"Successfully downloaded {package}")
                except Exception as e:
                    logger.error(f"Error downloading {package}: {str(e)}")
                    # Continue even if download fails, as the data might already be present
                    pass

    def generate_questions(self, content: str, num_questions: int = 10) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Generate questions from the given content.
        
        Args:
            content (str): The text content to generate questions from
            num_questions (int): Number of questions to generate
            
        Returns:
            List[Dict]: List of questions with their answers
        """
        sentences = sent_tokenize(content)
        questions = []
        
        # Generate fill-in-the-blank questions
        fill_blanks = self._generate_fill_in_blank(sentences, num_questions // 2)
        questions.extend(fill_blanks)
        
        # Generate multiple choice questions
        multiple_choice = self._generate_multiple_choice(sentences, num_questions - len(fill_blanks))
        questions.extend(multiple_choice)
        
        # Shuffle questions
        random.shuffle(questions)
        return questions[:num_questions]

    def _generate_fill_in_blank(self, sentences: List[str], num_questions: int) -> List[Dict]:
        """Generate fill-in-the-blank questions."""
        questions = []
        for sentence in sentences:
            if len(questions) >= num_questions:
                break
                
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            # Look for nouns or important terms to blank out
            for i, (word, tag) in enumerate(tagged):
                if (tag.startswith('NN') or tag.startswith('NNP')) and len(word) > 3 and word.lower() not in self.stop_words:
                    blank_sentence = ' '.join(words[:i] + ['_____'] + words[i+1:])
                    questions.append({
                        'type': 'fill_blank',
                        'question': blank_sentence,
                        'answer': word
                    })
                    break
        
        return questions

    def _generate_multiple_choice(self, sentences: List[str], num_questions: int) -> List[Dict]:
        """Generate multiple choice questions."""
        questions = []
        for sentence in sentences:
            if len(questions) >= num_questions:
                break
                
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            # Find important terms to ask about
            for word, tag in tagged:
                if (tag.startswith('NN') or tag.startswith('NNP')) and len(word) > 3 and word.lower() not in self.stop_words:
                    # Create a question about this term
                    question = f"What is {word}?"
                    
                    # Generate incorrect options from other sentences
                    other_terms = self._get_similar_terms(sentences, word)
                    if len(other_terms) >= 3:
                        options = random.sample(other_terms, 3)
                        options.append(word)
                        random.shuffle(options)
                        
                        questions.append({
                            'type': 'multiple_choice',
                            'question': question,
                            'options': options,
                            'answer': word
                        })
                        break
        
        return questions

    def _get_similar_terms(self, sentences: List[str], target_word: str) -> List[str]:
        """Get similar terms from other sentences to use as multiple choice options."""
        terms = set()
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            for word, tag in tagged:
                if (tag.startswith('NN') or tag.startswith('NNP')) and \
                   len(word) > 3 and \
                   word.lower() not in self.stop_words and \
                   word != target_word:
                    terms.add(word)
        
        return list(terms)

# Initialize the question generator
question_generator = QuestionGenerator() 