# Day 1 Corpus Audit

## Corpus status
This project uses a **synthetic internal-company corpus** because no real internal documents were available.

- Company: NovaTech Solutions
- Documents: 30 PDF files
- Format: PDF
- Pages per document: 1
- Total pages: 30
- Document types: HR, IT, Security, Finance, Operations, Legal, Communications, Administration
- Scanned/image-only documents: None
- Tables: None
- Golden questions: 30
- Answerable questions: 25
- Unanswerable questions: 5

## Evaluation design
The golden set was created before retrieval pipeline code. Every answerable question has an exact source filename and page number. The five unanswerable questions ask for information absent from the corpus.

## Limitation
The corpus is synthetic and must be described as such in the final README/report. It is intended to demonstrate the required RAG evaluation workflow when real internal documents are unavailable.
