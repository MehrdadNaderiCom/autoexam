from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch
import nltk
import random
from typing import List, Dict
import logging

class QuestionGenerator:
    def __init__(self):
        # Model for general question generation
        self.qg_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        
        logging.info("Question Generator initialized")

    def generate_multiple_choice_question(self, context: str) -> Dict:
        """Generate a multiple-choice question from the given context."""
        input_text = f"generate multiple choice question: {context}"
        
        # Generate question
        inputs = self.qg_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.qg_model.generate(
            inputs.input_ids,
            max_length=64,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True
        )
        
        question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Generate distractors (wrong answers)
        words = nltk.word_tokenize(context)
        pos_tags = nltk.pos_tag(words)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        
        # Create options including the correct answer
        correct_answer = context.split('.')[0]  # Use first sentence as correct answer
        options = [correct_answer]
        
        # Add 3 random nouns as distractors
        if len(nouns) >= 3:
            distractors = random.sample(nouns, 3)
            options.extend(distractors)
        else:
            # If not enough nouns, use parts of the context
            sentences = nltk.sent_tokenize(context)
            distractors = random.sample(sentences, min(3, len(sentences)))
            options.extend(distractors)
        
        random.shuffle(options)
        
        return {
            'question_text': question,
            'answer_text': correct_answer,
            'options': options,
            'question_type': 'multiple_choice',
            'source_text': context
        }

    def generate_fill_blank_question(self, context: str) -> Dict:
        """Generate a fill-in-the-blank question from the given context."""
        # Find important nouns in the context
        words = nltk.word_tokenize(context)
        pos_tags = nltk.pos_tag(words)
        nouns = [(word, i) for i, (word, pos) in enumerate(pos_tags) if pos.startswith('NN')]
        
        if not nouns:
            return None
            
        # Select a random noun to blank out
        word_to_blank, index = random.choice(nouns)
        
        # Create the fill-in-the-blank question
        words[index] = "________"
        question_text = " ".join(words)
        
        return {
            'question_text': question_text,
            'answer_text': word_to_blank,
            'question_type': 'fill_blank',
            'source_text': context
        }

    def generate_questions(self, text_chunks: List[str], num_questions: int = 10) -> List[Dict]:
        """
        Generate a mix of multiple-choice and fill-in-the-blank questions.
        """
        questions = []
        num_each_type = num_questions // 2
        
        # Generate multiple-choice questions
        for chunk in text_chunks[:num_each_type]:
            try:
                question = self.generate_multiple_choice_question(chunk)
                if question:
                    questions.append(question)
            except Exception as e:
                logging.error(f"Error generating multiple-choice question: {str(e)}")
                continue
        
        # Generate fill-in-the-blank questions
        for chunk in text_chunks[num_each_type:num_questions]:
            try:
                question = self.generate_fill_blank_question(chunk)
                if question:
                    questions.append(question)
            except Exception as e:
                logging.error(f"Error generating fill-in-blank question: {str(e)}")
                continue
        
        return questions[:num_questions]

# Initialize the question generator
question_generator = QuestionGenerator() 