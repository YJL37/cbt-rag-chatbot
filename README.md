# CBT RAG Chatbot
Counseling Chatbot for Cognitive Behavioral Therapy (CBT)

## Overview
As teenage anxiety and depression rise, driven in part by social media, the research aims to develop a Retrieval-Augmented Generation chatbot for Cognitive Behavioral Therapy to provide accessible support. We integrated three CBT-related datasets using LangChain and Google's Gemini to generate personalized counseling responses. Our evaluation showed high answer relevancy and perfect faithfulness scores, confirming that the chatbot is adept at answer generation. However, low contextual recall suggests improvements are needed for better context retrieval. The chatbot demonstrates strong potential in addressing mental health issues among teenagers, with plans for further enhancements with diary input and graph database to improve response accuracy.

## Tech Stacks
- **Language**: Python
- **Large Language Model (LLM)**: Gemini-1.5-Flash
- **Framework**: LangChain, LangGraph
- **Database**: ChromaDB (VectorDB), Neo4j (Knowledge Graph)

## Getting Started
Install packages
'''bash
pip install -r requirements.txt
'''

Run Python script for CBT RAG CLI
'''bash
python run.py --eval
'''

