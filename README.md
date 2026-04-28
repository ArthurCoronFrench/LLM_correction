# Projet début LLM correction grammaire

Un projet de correction grammaticale automatique en espagnol basé sur le deep learning, construit avec PyTorch.
Cette version propose une architecture propre et compacte (6 fichiers) tout en couvrant l’ensemble du pipeline NLP :
 - génération de dataset
 - preprocessing
 - modèles (LSTM + Attention + Seq2Seq)
 - entraînement
 - inference (correction)

# Objectif

Créer un système capable de détecter des erreurs grammaticales, comprendre le contexte d’une phrase et de corriger automatiquement la phrase complète.

## 1. Dataset 
Génération automatique d'un dataset de manière dynamique (sujets, verbes conjugués, objets, adjectifs, compléments).
Chaque exemple contient un input incorrect et un output corrigé.
## 2. Preprocessing 
### Etape 1: Construction du vocabulaire
word_to_idx = build_vocab(dataset)
→ transforme les mots en indices
### Etape 2: Encodage
transforme chaque mot en vecteurs
### Etape 3 Padding 
toutes les phrases sont normalisées à une longueur fixe
## 3. Models
On regroupe tous les modèles utilisés dans ce code.

 - POS tagger: classification des mots
 - Error model: détection d'erreur (en binaire)
 - Seq2seq: encoder + attention + decoder
## 4. Pipeline
Entrainement et inférence
