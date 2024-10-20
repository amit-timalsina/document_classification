# Document Classification

## Template Matching
Write rules for template matching

Document -> OCR -> Data Preparation -> Template Matching

Manual human intervention required. But can be automated by asking LLM to write rules. We will try both (manual and using LLM).

## Fasttext
Training
Document -> OCR -> Train / Val / Test data -> Fasttext training

Inference
Document -> OCR -> Data preparation -> Fasttext inference

## BERT
Training
Document -> OCR -> Train / Val / Test data -> BERT training

Inference
Document -> OCR -> Data preparation -> BERT inference

## Vision models
I don't know how this works

## Text + Layout Models (BROS, LayoutLM)

## Text + Layout Models + Vision Models (LayoutLMV2, etc)

## Large Language models (LLMs)

## Semantic Similarity

## Clustering

# OCR

We need to give option to use both open source models like:
- Teserract
- PaddleOCR

and api providers like:
- Google Vision API
- AWS Textract
- Azure OCR

Furthermore, we might want to give support to use in built ocr engine like in case od Donut model
Also, support to load any existing model in Huggigface liek GOT, etc.