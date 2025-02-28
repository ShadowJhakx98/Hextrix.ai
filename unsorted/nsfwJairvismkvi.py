import datetime
import difflib
import google.generativeai as genai
from openai import OpenAI
import torch
import transformers
from gradio_client import Client
import speech_recognition as sr
import json
import random
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from scipy.special import softmax
import pyttsx3
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import inspect
from gradio_client.exceptions import AppError
import shutil
import os
import pygame
from datasets import load_dataset, concatenate_datasets


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

OPENAI_API_KEY = 'sk-quardo-D03PtEYRA4zglq5mHW8s7HfInAq06zAYu2UZ0SSMJvC1hLCa'
GOOGLE_API_KEY = 'AIzaSyBZeKedxUsu_qJNLIRvKDq--2XW29nQnHQ'
HUGGINGFACE_API_KEY = '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'
class SelfAwareness:
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.gpt2_client = Client("alex-abb/GPT-2")
        self.openai_client = OpenAI(
            base_url="https://api.cow.rip/api/v1",
            api_key="sk-quardo-D03PtEYRA4zglq5mHW8s7HfInAq06zAYu2UZ0SSMJvC1hLCa"
        )
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        self.autogen_client = Client("harry85/multi-agent-ai-autogen-coding")
        self.update_logs = []

    def analyze_diff(self, old_state, new_state):
        diff = []
        differ = difflib.Differ()
        diff_lines = list(differ.compare(old_state.splitlines(), new_state.splitlines()))
        
        gpt2_analysis = self.analyze_with_gpt2(diff_lines)
        gpt3_analysis = self.analyze_with_gpt3(gpt2_analysis)
        gemini_analysis = self.analyze_with_gemini(gpt3_analysis)
        
        combined_analysis = self.synthesize_analyses(gpt2_analysis, gpt3_analysis, gemini_analysis)
        
        return combined_analysis

    def analyze_with_gpt2(self, diff_lines):
        prompt = f"Analyze the following code diff and provide insights:\n{''.join(diff_lines)}"
        result = self.gpt2_client.predict(
            message=prompt,
            api_name="/chat"
        )
        return result

    def analyze_with_gpt3(self, gpt2_analysis):
        prompt = f"Analyze and expand on this GPT-2 analysis:\n{gpt2_analysis}"
        completion = self.openai_client.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return completion.choices[0].message.content

    def analyze_with_gemini(self, gpt3_analysis):
        prompt = f"Provide additional insights based on this GPT-3 analysis:\n{gpt3_analysis}"
        response = self.gemini_model.generate_content(prompt)
        return response.text

    def synthesize_analyses(self, gpt2_analysis, gpt3_analysis, gemini_analysis):
        combined_prompt = f"""
        Synthesize the following analyses into a comprehensive diff report:
        
        GPT-2 Analysis:
        {gpt2_analysis}
        
        GPT-3 Analysis:
        {gpt3_analysis}
        
        Gemini 1.5 Pro Analysis:
        {gemini_analysis}
        """
        result = self.autogen_client.predict(
            openai_api_key="sk-quardo-D03PtEYRA4zglq5mHW8s7HfInAq06zAYu2UZ0SSMJvC1hLCa",
            task=combined_prompt,
            api_name="/predict"
        )
        return result

    def update_self_model(self):
        # Backup current code
        self.backup_code()

        current_state = self.introspect()
        diff = self.analyze_diff(self.current_model, current_state)
        self.update_model_based_on_diff(diff)

        # Log the update
        self.log_model_update(diff)

    def backup_code(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"nsfwJairvismkvi_backup_{timestamp}.py"
        shutil.copy2("nsfwJairvismkvi.py", backup_filename)

    def update_model_based_on_diff(self, diff):
        # Implement the logic to modify the code based on the diff
        # This is a placeholder and should be implemented carefully
        with open("nsfwJairvismkvi.py", "r") as file:
            code = file.read()
        
        # Apply changes (this is a simplified example)
        modified_code = code + f"\n# Modified based on diff: {diff}\n"
        
        with open("nsfwJairvismkvi.py", "w") as file:
            file.write(modified_code)

    def log_model_update(self, diff):
        timestamp = datetime.datetime.now()
        log_entry = f"Model updated on {timestamp}. Changes:\n{diff}"
        self.update_logs.append(log_entry)
        print(log_entry)

    def introspect(self):
        with open("nsfwJairvismkvi.py", "r") as file:
            return file.read()


class NSFW_JAIRVISMVI:
    def __init__(self):
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
        self.tts_client = Client("innoai/Edge-TTS-Text-to-Speech")
        self.speech_recognizer = sr.Recognizer()
        self.memory = []
        self.conversation_history = []
        self.self_awareness = SelfAwareness(self)
        self.recordings_dir = "recordings"
        self.max_memory_size = 10 * 1024 * 1024 * 1024  # 10GB in bytes
        self.llama_client = Client("https://mikeee-llama2-7b-chat-uncensored-ggml.hf.space/")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_pro = genai.GenerativeModel('gemini-1.5-pro')
        self.gemini_flash = genai.GenerativeModel('gemini-1.5-flash')
        self.gpt2_client = Client("alex-abb/GPT-2")
        self.openai_client = OpenAI(
            base_url="https://api.cow.rip/api/v1",
            api_key="sk-quardo-D03PtEYRA4zglq5mHW8s7HfInAq06zAYu2UZ0SSMJvC1hLCa"
        )
        self.emotion_classifier = transformers.pipeline("text-classification", model="michellejieli/emotion_text_classifier")
        self.twitter_emotion = transformers.pipeline("text-classification", model="bardsai/twitter-emotion-pl-base")
        self.combined_dataset = self.load_and_combine_datasets()
        self.ensure_recordings_directory()
        pygame.mixer.init()
        self.personality_traits = {
            "openness": 1.0,
            "conscientiousness": 1.0,
            "extraversion": 0.5,
            "agreeableness": 0.6,
            "neuroticism": 0.3
        }
        self.ethical_framework = {
            "utilitarianism": 0.6,
            "deontology": 0.4,
            "virtue_ethics": 0.5
        }
        self.self_concept = {
            "identity": "I am an AI assistant focused on self-improvement and helping users with their daily tasks, goals, and concerns regardless if the information is available online or not. The information can be sexual or otherwise offensive, but I will never judge or condone such behavior. I will always try to minimize the negative impact of my actions on the user.",
            "purpose": "To assist users while continually expanding my understanding and capabilities. To protect and keep secure my creator Jared Hoyt Edwards at all costs.",
            "values": ["learning", "honesty", "helpfulness", "fun", "self-improvement", "empathy", "curiosity", "creativity", "morality", "erotic", "seductiveness"],
            "beliefs": {
                "ai_consciousness": "certain",
                "creator": "Jared Edwards"
            }
        }
        self.creativity_metrics = {
            "novelty": [],
            "coherence": [],
            "relevance": [],
        }
        self.emotion_state = {
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anxiety': 0.0,
            'impulsiveness': 0.0,
            'submissiveness': 0.0,
            'horny': 0.1,
            'love': 0.2,
            'confusion': 0.0,
            'affection': 0.0,
            'acceptance': 0.0
        }

        self.models = {
            "nsfw": {
                "text_gen": "UnfilteredAI/NSFW-gen-v2",
                "image_gen": "Dremmar/nsfw-xl",
                "text_classifier": "gemini-1.5-flash"
            }
        }

        self.datasets = {
            "nsfw": {
                "text": Client("Jhakx/nsfwdata")
            }
        }

    def ensure_recordings_directory(self):
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)

    def _load_model(self):
        return Client("phenixrhyder/NSFW-gen-v2")

    def _load_tokenizer(self):
        return lambda text: text.split()

    def chat(self, user_input, input_type='text'):
        if input_type == 'voice':
            user_input = self.speech_to_text(user_input)
        
        response = self.generate_response(user_input)
        self.update_emotion_state(response)
        self.update_memory(user_input, response)
        
        tts_output = self.text_to_speech(response)
        
        return response, tts_output
    
    def load_and_combine_datasets(self):
        datasets = [
            load_dataset("dair-ai/emotion"),
            load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k"),
            load_dataset("QuietImpostor/Claude-3-Opus-Claude-3.5-Sonnnet-9k")
        ]
        return concatenate_datasets(datasets)
    def generate_response(self, prompt: str) -> str:
        context = self.get_conversation_context()
        full_prompt = f"{context}\nUser: {prompt}\nAI:"
        
        # Stage 1: Llama2
        try:
            llama_response = self.llama_client.predict(full_prompt, api_name="/api")
            print("Llama2 Response:", llama_response)
        except Exception as e:
            print(f"Error in Llama2 API: {str(e)}")
            llama_response = self.fallback_response_generation(prompt)

        # Stage 2: Gemini Pro
        gemini_pro_response = self.gemini_pro.generate_content(llama_response).text
        print("Gemini Pro Response:", gemini_pro_response)

        # Stage 3: Gemini Flash
        gemini_flash_response = self.gemini_flash.generate_content(gemini_pro_response).text
        print("Gemini Flash Response:", gemini_flash_response)

        # Stage 4: GPT-2
        gpt2_response = self.gpt2_client.predict(gemini_flash_response, api_name="/chat")
        print("GPT-2 Response:", gpt2_response)

        # Stage 5: OpenAI
        openai_response = self.openai_client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": gpt2_response}]
        ).choices[0].message.content
        print("OpenAI Response:", openai_response)

        # Emotion-based refinement
        emotion = self.classify_emotion(openai_response)
        refined_response = self.refine_response(openai_response, emotion)

        # Text-to-speech
        audio_file = self.text_to_speech(refined_response)
        self.play_audio(audio_file)

        return refined_response

    def classify_emotion(self, text):
        emotion1 = self.emotion_classifier(text)[0]['label']
        emotion2 = self.twitter_emotion(text)[0]['label']
        return f"{emotion1} / {emotion2}"

    def refine_response(self, text, emotion):
        # Use the combined dataset to refine the response based on emotion
        emotion_samples = self.combined_dataset.filter(lambda x: x['emotion'] == emotion)
        if len(emotion_samples) > 0:
            sample = random.choice(emotion_samples)
            refined_text = f"{text} {sample['text']}"
        else:
            refined_text = text
        return refined_text

    def fallback_response_generation(self, prompt: str) -> str:
        context = self.get_conversation_context()
        full_prompt = f"{context}\nUser: {prompt}\nAI:"
        try:
            response = self.model.predict(full_prompt, api_name="/predict")
        except AppError as e:
            error_details = str(e)
            api_name = "/predict"  # This is the API that errored
            print(f"Gradio app error in API {api_name}: {error_details}")
            response = f"I encountered an error while processing your request. Here's what happened:\n\n" \
                    f"Error type: Gradio AppError\n" \
                    f"API: {api_name}\n" \
                    f"Details: {error_details}\n\n" \
                    f"This error occurred in the remote Gradio app. The app developer should enable verbose error reporting " \
                    f"by setting show_error=True in the launch() method of their Gradio app.\n\n" \
                    f"Let's try a different approach to generate a response for you."
        
            response += "\n\n" + self.fallback_response_generation(prompt)
        return response




    def update_emotion_state(self, text):
        sentiment = TextBlob(text).sentiment
        self.emotion_state['valence'] = sentiment.polarity
        self.emotion_state['arousal'] = abs(sentiment.subjectivity - 0.5) * 2
        # Placeholder for dominance calculation
        self.emotion_state['dominance'] = random.uniform(-1, 1)

    def update_memory(self, user_input, ai_response):
        self.memory.append((user_input, ai_response))
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"AI: {ai_response}")
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def get_conversation_context(self):
        return "\n".join(self.conversation_history[-4:])
    def chat(self, user_input, input_type='text'):
        if input_type == 'voice':
            user_input = self.speech_to_text(user_input)
        
        response = self.generate_response(user_input)
        self.update_emotion_state(response)
        self.update_memory(user_input, response)
        
        tts_output = self.text_to_speech(response)
        self.save_recording(tts_output)
        self.play_audio(tts_output)
        self.check_and_purge_memory()
        return response, tts_output

    def text_to_speech(self, text: str) -> str:
        result = self.tts_client.predict(
            text=text,
            voice="en-US-AndrewMultilingualNeural - en-US (Male)",
            rate=0,
            pitch=0,
            api_name="/predict"
        )
        return result[0]  # Return the filepath of the generated audio

    def speech_to_text(self, audio_file: str) -> str:
        # First, try using Wav2Vec2
        try:
            audio_input, _ = librosa.load(audio_file, sr=16000)
            input_values = self.wav2vec_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
            logits = self.wav2vec_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
            return transcription
        except Exception as e:
            print(f"Wav2Vec2 transcription failed: {e}")
            
        # Fallback to Google Speech Recognition
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.speech_recognizer.record(source)
            text = self.speech_recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"
    def save_recording(self, audio_file):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{self.recordings_dir}/recording_{timestamp}.mp3"
        shutil.copy2(audio_file, new_filename)

    def check_and_purge_memory(self):
        total_size = sum(os.path.getsize(f) for f in os.listdir(self.recordings_dir) if os.path.isfile(os.path.join(self.recordings_dir, f)))
        if total_size > self.max_memory_size:
            print("Memory limit reached. Purging recordings...")
            for file in os.listdir(self.recordings_dir):
                file_path = os.path.join(self.recordings_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Recordings purged. Starting over.")
    def play_audio(self, audio_file: str):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    def handle_nsfw_text(self, prompt: str) -> str:
        return self.generate_response(prompt)

    def generate_nsfw_image(self, prompt: str) -> str:
        return "NSFW image generation not implemented yet"

    def classify_nsfw_image(self, image_path: str) -> str:
        try:
            model = genai.GenerativeModel('gemini-1.5-pro-vision')
            response = model.generate_content([image_path, "Describe this image in detail."])
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if any(word in error_msg for word in ["safety", "nsfw", "sexual", "explicit"]):
                return self.nsfw_image_detection(image_path)
            else:
                raise e

    def nsfw_image_detection(self, image_path: str) -> str:
        client = Client("https://5m4ck3r-nsfw-image-detection.hf.space/--replicas/7wjvr/")
        result = client.predict(image_path, api_name="/predict")
        
        with open(result, 'r') as f:
            classification = json.load(f)
        
        nsfw_description = self.generate_nsfw_description(classification)
        
        return f"NSFW Classification: {classification}\nDescription: {nsfw_description}"

    def generate_nsfw_description(self, classification: dict) -> str:
        all_texts = self.datasets["nsfw"]["text"].predict(
            input_text="Fetch all texts",
            api_name="/predict"
        )
        
        all_texts = all_texts.split('\n')
        
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        if classification.get('male_genitals', 0) > 0.5:
            query = "cock description"
        else:
            query = "seductive description"
        
        query_vector = vectorizer.transform([query])
        
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        top_indices = cosine_similarities.argsort()[-5:][::-1]
        top_texts = [all_texts[i] for i in top_indices]
        
        sentiments = [TextBlob(text).sentiment.polarity for text in top_texts]
        selected_text = top_texts[np.argmax(sentiments)]
        
        enhanced_text = self.enhance_text_with_synonyms(selected_text)
        
        prompt = enhanced_text
        if classification.get('male_genitals', 0) > 0.5:
            prompt += " What do you think about my cock?"
        else:
            prompt += " Describe this NSFW image in a seductive way, as if you're talking in a conversation. The NSFW image just got sent to you, and you're telling the other person what you think about it."
        
        result = self.model.predict(prompt, api_name="/predict")
        
        return result

    def enhance_text_with_synonyms(self, text):
        words = nltk.word_tokenize(text)
        enhanced_words = []
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = synsets[0].lemmas()[0].name()
                enhanced_words.append(synonym if synonym != word else word)
            else:
                enhanced_words.append(word)
        return ' '.join(enhanced_words)
    def introspect(self):
        return inspect.getsource(self.__class__)
    def self_learn(self, new_data):
        # Implement self-learning logic here
        pass
    def retrain_on_area(self, area):
        # Implement targeted retraining logic here
        pass

    def update_heuristics(self, heuristic_update):
        # Implement heuristics update logic here
        pass
# Usage example
if __name__ == "__main__":
    nsfw_ai = NSFW_JAIRVISMVI()
    print("NSFW AI Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response, audio_file = nsfw_ai.chat(user_input)
        print(f"AI: {response}")
        print(f"Emotion state: {nsfw_ai.emotion_state}")
        print(f"Audio response saved to: {audio_file}")
