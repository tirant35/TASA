# Data in experiment
All the data used in the experiment, including medical, legal and financial data, came from medQA, BioInstruct, FinGPT, Legalbench, etc
- finGPT：https://github.com/AI4Finance-Foundation/FinGPT https://huggingface.co/FinGPT
    - Sentiment Analysis
    - Relation Extraction
    - Headline Analysis
    - Named-Entity Recognition
    - Question Answer
    - Chinese Multiple-Choice Questions
- BioInstruct：https://github.com/bio-nlp/BioInstruct https://huggingface.co/datasets/bio-nlp-umass/bioinstruct
    - Information Extraction
    - Question Answer
    - Text Generation
    - Other Tasks
- med_qa：https://huggingface.co/datasets/bigbio/med_qa
    - multiple choice questions
- legalbench：https://github.com/HazyResearch/legalbench/ https://huggingface.co/datasets/nguha/legalbench/blob/main/legalbench.py https://huggingface.co/datasets/Equall/legalbench_instruct
    - multiple choice questions
    - Judging Yes or No
    - Judging Correct or Incorrect
    - Judging Relevant or Irrelevant
    - Information Extraction
    - Definition Extraction
    - Text Classification
### 1. `data_adapters`: data for training and testing the single-task best practice adapter in the domain


### 2. `data_selector`: data for training the Selector

### 3. `test`: data for testing the Selector
