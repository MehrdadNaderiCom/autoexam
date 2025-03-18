from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QuestionGenerator:
    def __init__(self):
        self.model_name = "valhalla/t5-small-qg-hl"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        logging.info(f"Loaded model: {self.model_name}")

    def generate_questions(self, text: str, max_questions: int = 5) -> list:
        """
        Generate questions from the input text.
        Args:
            text: Input text to generate questions from
            max_questions: Maximum number of questions to generate
        Returns:
            List of dictionaries containing questions and answers
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        questions = []
        
        for sentence in sentences[:max_questions]:  # Limit to max_questions
            try:
                # Prepare input for the model
                input_text = f"generate question: {sentence}"
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

                # Generate question
                outputs = self.model.generate(
                    input_ids,
                    max_length=64,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )

                question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                questions.append({
                    'question_text': question,
                    'answer_text': sentence,
                    'question_type': 'short_answer',
                    'source_text': text
                })

            except Exception as e:
                logging.error(f"Error generating question for sentence: {sentence}")
                logging.error(str(e))
                continue

            if len(questions) >= max_questions:
                break

        return questions

# Initialize the question generator
question_generator = QuestionGenerator() 