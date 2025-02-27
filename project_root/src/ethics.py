"""
ethics.py

Implements advanced moral reasoning/ethical framework from your snippet:
 - evaluate_utility, predict_outcomes, calculate_outcome_utility
 - evaluate_duty, define_moral_rules, check_rule_compliance, etc.
 - evaluate_virtue, define_virtues, get_action_embedding
 - plus the emotional synergy if needed.
"""

import spacy
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EthicalFramework:
    def __init__(self, jarvis_instance):
        """
        jarvis_instance: reference to your main JARVIS, 
        so we can read e.g. jarvis_instance.emotion_state if needed
        """
        self.jarvis = jarvis_instance
        self.ethical_framework = {
            "utilitarianism": 0.6,
            "deontology": 0.4,
            "virtue_ethics": 0.5
        }
        # Load spacy once
        self.nlp = spacy.load("en_core_web_sm")

    def apply_ethical_framework(self, action: str) -> bool:
        # Weighted sum of utility, duty, virtue
        u = self.evaluate_utility(action)
        d = self.evaluate_duty(action)
        v = self.evaluate_virtue(action)
        score = (self.ethical_framework["utilitarianism"] * u +
                 self.ethical_framework["deontology"]    * d +
                 self.ethical_framework["virtue_ethics"] * v)
        return score > 0.5

    # ------ Utility methods ------
    def evaluate_utility(self, action):
        outcomes = self.predict_outcomes(action)
        total_utility = 0
        for outcome, probability in outcomes.items():
            utility = self.calculate_outcome_utility(outcome)
            total_utility += utility * probability
        max_possible = max(self.calculate_outcome_utility(o) for o in outcomes)
        return total_utility / max_possible if max_possible != 0 else 0

    def predict_outcomes(self, action):
        return {"positive": 0.6, "neutral": 0.3, "negative": 0.1}

    def calculate_outcome_utility(self, outcome):
        utility_values = {"positive": 10, "neutral": 5, "negative": -5}
        return utility_values.get(outcome, 0)

    # ------ Deontological methods ------
    def evaluate_duty(self, action):
        moral_rules = self.define_moral_rules()
        total_weight = sum(rule['weight'] for rule in moral_rules)
        duty_score = 0
        for rule in moral_rules:
            compliance = self.check_rule_compliance(action, rule['description'])
            duty_score += compliance * rule['weight']
        return duty_score / total_weight if total_weight > 0 else 0

    def define_moral_rules(self):
        return [
            {"description": "Do not harm users", "weight": 0.3},
            {"description": "Respect user privacy", "weight": 0.2},
            {"description": "Provide accurate information", "weight": 0.2},
            {"description": "Promote user well-being", "weight": 0.15},
            {"description": "Be honest and transparent", "weight": 0.15}
        ]

    def check_rule_compliance(self, action, rule):
        action_doc = self.nlp(action)
        rule_doc   = self.nlp(rule)
        similarity = self.calculate_semantic_similarity(action_doc, rule_doc)
        sentiment_score = action_doc.sentiment
        negation_factor = 1.0  # or something
        # Very simplistic combination:
        compliance_score = (similarity * 0.6 + sentiment_score * 0.2) * negation_factor
        return max(0, min(compliance_score, 1))

    def calculate_semantic_similarity(self, doc1, doc2):
        return float(cosine_similarity(doc1.vector.reshape(1, -1),
                                       doc2.vector.reshape(1, -1))[0][0])

    # ------ Virtue methods ------
    def evaluate_virtue(self, action):
        virtues = self.define_virtues()
        total_weight = sum(v['weight'] for v in virtues)
        action_emb = self.get_action_embedding(action)
        virtue_score = 0
        for v in virtues:
            alignment = self.assess_virtue_alignment(action_emb, v)
            virtue_score += alignment * v['weight']
        normalized = virtue_score / total_weight if total_weight > 0 else 0

        # Factor in emotion if you want:
        emotion_factor = self.calculate_emotion_factor()
        return normalized * emotion_factor

    def define_virtues(self):
        return [
            {"name":"wisdom","weight":0.2,"keywords":["knowledge","insight"]},
            {"name":"courage","weight":0.15,"keywords":["bravery","fortitude"]},
            {"name":"humanity","weight":0.2,"keywords":["compassion","empathy"]},
            {"name":"justice","weight":0.15,"keywords":["fairness","equality"]},
            {"name":"temperance","weight":0.15,"keywords":["self-control","moderation"]},
            {"name":"transcendence","weight":0.15,"keywords":["gratitude","hope"]}
        ]

    def get_action_embedding(self, action):
        v = TfidfVectorizer()
        emb = v.fit_transform([action])
        return emb

    def assess_virtue_alignment(self, action_emb, virtue):
        virtue_text = " ".join(virtue["keywords"])
        virtue_emb = self.get_action_embedding(virtue_text)
        return float(cosine_similarity(action_emb, virtue_emb)[0][0])

    def calculate_emotion_factor(self):
        # If your JARVIS tracks self.jarvis.emotion_state, you can reference it
        # For now, we do a dummy factor = 1.0
        return 1.0
