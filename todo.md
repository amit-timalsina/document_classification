# Usage Rules:
# TODO file based on the todo.txt format - https://github.com/todotxt/todo.txt
# Use the todo.txt format for the todo list
# Please get the `usernamehw.todo-md` extension for the vscode.
# Use the following tags: #todo, #in-progress, #done, #blocked, #urgent.
# Please use Asana for the project management and avoid this file.
# This file is only meant for non-important tasks like: use AwareDatetime instead of Datetime, etc.
# If this file grows before 100 tasks, move everything to Asana and trim this file back to zero.
# Please add Assignee to each task.
# It is always better to add todos next to the code changes.
# Only add things here when not possible to add next to the code changes.

OCR
    Open Source
        Tesseract {cm:2024-10-21}
        Paddle
    Closed Source
        Google OCR {cm:2024-10-21}
        AWS Textract
        Azure
    Hugging Face
        Donut
        GOT
        Others

Classification
    Text only Models
        LLM Based classification {due:2024-10-22}
        Fasttext {due:2024-10-22}
        BERT
    Text + Layout Models
        BROS
        LayoutLM {due:2024-10-23}
    Text + Layout Models + Vision Models
        LayoutLMV2
        Donut {due:2024-10-23}
    Embedding
        Semantic Similarity
        Clustering
    Vision only Models