# Document Classification: All in one place
This package provides support to classify documents using many popular methods. Along with document classification, it also provides support to a single interface for OCR using both open source models like: Tesseract and PaddleOCR, and commercial models like Google OCR, etc.

PYPI: [document-classification](https://pypi.org/project/document-classification/)

## Features
- OCR
    - Tesseract
    - Google OCR
- Classification
    - Fasttext (train, evaluate, predict)
    - Language Models like BERT (train, evaluate, predict)
    - Language + Layout Models like LayoutLM (train, evaluate, predict)
    - LLM (evaluate, predict)

## Installation
Install with a single command:
```bash
pip install -U document-classification
```
or if you use poetry (like me):
```bash
poetry add document-classification
```

## Usuage
Please check the [examples](https://github.com/amit-timalsina/document_classification/tree/master/examples) directory for examples on how to use the package.

## Contributing

Your contributions are welcome! If you have great examples or find neat patterns, clone the repo and add another example. 
The goal is to find great patterns and cool examples to highlight.

If you encounter any issues or want to provide feedback, you can create an issue in this repository. You can also reach out to me on Twitter at [@amittimalsina14](https://x.com/amittimalsina14).

Check the [todo.md](https://github.com/amit-timalsina/document_classification/blob/master/todo.md) file for the list of features that are coming next with their due dates.

## What's coming next?
I am going to first add tests and refactor the code to make it more readable, usuable, and maintainable. Then I will release documentation and more examples.
