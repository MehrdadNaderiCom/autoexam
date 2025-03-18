import os
import random
import nltk
import logging
import openai
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator."""
        self.use_basic_tokenization = True
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/averaged_perceptron_tagger')
            self.use_basic_tokenization = False
            logger.info("NLTK data found, using advanced tokenization")
        except LookupError:
            logger.info("NLTK data not found, using basic tokenization")

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Split on common sentence endings
        text = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|')
        sentences = text.split('|')
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into sentences using either NLTK or basic tokenization."""
        try:
            if not self.use_basic_tokenization:
                return nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
        return self._basic_tokenize(text)

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract potential keywords from a sentence."""
        try:
            if not self.use_basic_tokenization:
                words = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(words)
                return [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        except Exception as e:
            logger.warning(f"NLTK keyword extraction failed: {str(e)}")
        
        # Fallback: return words with 4 or more characters
        words = sentence.split()
        return [word for word in words if len(word) >= 4 and word.isalnum()]

    def _enhance_with_chatgpt(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """Use ChatGPT to create a better question from the sentence and keyword."""
        try:
            if not openai.api_key:
                logger.warning("OpenAI API key not found, falling back to basic question generation")
                return None

            prompt = f"""Given this sentence: "{sentence}"
            Create a multiple-choice question that tests understanding of the concept around the word "{keyword}".
            Format your response as a JSON object with these fields:
            - question: The question text
            - correct_answer: The correct answer
            - options: Array of 4 options including the correct answer
            - explanation: Brief explanation of why the correct answer is right
            
            Make the question professional and meaningful, testing real understanding rather than just memorization."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )

            # Parse the response
            content = response.choices[0].message.content
            try:
                # Try to evaluate the JSON-like string (safer than eval)
                import ast
                result = ast.literal_eval(content)
                return {
                    'type': 'multiple_choice',
                    'question': result['question'],
                    'options': result['options'],
                    'answer': result['correct_answer'],
                    'explanation': result['explanation']
                }
            except:
                logger.warning("Failed to parse ChatGPT response, falling back to basic question")
                return None

        except Exception as e:
            logger.warning(f"ChatGPT enhancement failed: {str(e)}")
            return None

    def _generate_basic_question(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a basic fill-in-the-blank question as fallback."""
        question_text = sentence.replace(keyword, "________")
        return {
            'type': 'fill_blank',
            'question': question_text,
            'answer': keyword,
            'options': [],
            'explanation': f"The word '{keyword}' fits in this context based on the original text."
        }

    def generate_questions(self, content, num_questions=5):
        """Generate questions from the given content."""
        try:
            sentences = self._tokenize_text(content['text'])
            if not sentences:
                return None
            
            questions = []
            wiki_url = content.get('url', '')  # Get Wikipedia URL from content
            
            for _ in range(min(num_questions, len(sentences))):
                sentence = random.choice(sentences)
                sentences.remove(sentence)  # Avoid reusing the same sentence
                
                try:
                    keywords = self._extract_keywords(sentence)
                    if keywords:
                        keyword = random.choice(keywords)
                        question_data = self._enhance_with_chatgpt(sentence, keyword)
                        if question_data:
                            question_data['source_url'] = wiki_url  # Add Wikipedia URL to question data
                            questions.append(question_data)
                except Exception as e:
                    logger.error(f"Error generating question with ChatGPT: {e}")
                    # Fallback to basic question if ChatGPT fails
                    question = self._generate_basic_question(sentence, keyword)
                    if question:
                        question['source_url'] = wiki_url  # Add Wikipedia URL to fallback question
                        questions.append(question)
            
            return questions if questions else None

# Initialize the question generator
question_generator = QuestionGenerator() 