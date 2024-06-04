The recent surge in Large Language Models (LLMs) has catalyzed significant advancements across various fields, with software engineering being no exception. Among these advancements, Retrieval Augmented Generation (RAG) is a particularly-appealing one. RAG systems combine retrieval of artifacts from a local (e.g., domain-specific) knowledge-basewith a LLM-based generation. This helps generating solutions to problems for which the LLM is not specifically trained or fine-tuned.One risk RAG may have is that, if the knowledge-basecontains noisy data, the produced output can be erroneous too.
This paper explores the resilience of RAG systems in the context of a software engineering task, specifically code summarization. We investigate what happens when RAG knowledge-basecontains various levels of data inconsistency/noisiness. We first assess the impact of inconsistent data to the retriever component implemented with two different techniques (BM25 or Transformer-based), and then the effect on the generator side, by leveraging  two state-of-the-art LLMs, CodeLlama and GPT-4. Our results show that, while Transformer-based retrieval methods generally outperform BM25 in terms of resilience, they struggle to maintain adherence to developer-written summaries as inconsistency levels in the knowledge-basefluctuate. On the generator side, both LLMs—CodeLlama and GPT-4—show resilience against varying levels of inconsistency, with CodeLlama displaying superior overall performance by guaranteeing that the generated code summaries are of higher quality. 

#### Pipeline Description

To replicate the experiments you can rely on this two files *RAG.py* and *gpt4-cs.py* under Code/.
While the former can be used to evaluate CodeLlama, the latter implements the API calls to interact with GPT4-o. Before starting replicating any of the experiments we performed, make sure to install the requirements (see *requirements.txt*)


#### Run RAG.py :computer:
For instance, to run CodeLlama7B against when the knowledge-base contains ~10% of inconsistent example, you first load all the documents (i.e., knowledge-base) using the script Code/loadDocs.py, then the following:

```python3 loadDocs.py -f Data/RAG-Inconsistency/inc-full-kb.csv```

then

```python3 RAG.py -m CodeLlama-7b -f Data/RAG-Inconsistency/inc-full-kb.csv```


##### Run gpt4-cs.py :computer:
Here's how to run resilience experiments with GPT4-o:

1) Generate prompts: Before testing GPT4-o, you'll need prompts to guide the model. These prompts can be automatically generated by running CodeLlama first.

2) Use the output file: Once CodeLlama finishes, the script will output a file with a new "prompt" column added specifically for testing GPT4-o. This file can be used directly for further testing with GPT4-o.

#### Additional Scripts:  :computer:

* <a href="https://drive.google.com/drive/folders/1ojCdNQAk1VNc1rriGIafFi1udIDvP9qw?usp=sharing">Scripts to conduct the statistical tests </a>

#### Datasets :paperclip:

* The test dataset for testing the resilience can be found under Data/test.csv
* The data for populating the different knowledge-bases are available under Data/Rag-Baseline and Data/Rag-Inconsistency

#### Additional Data:  :paperclip:

* <a href="https://drive.google.com/drive/folders/168CyI8VVN_hw3OxYHQd4I7uHh2XkNSsl?usp=share_link">Data for Statistical Tests (i.e., McNemar and Wilcoxon)</a>


#### Results:  :open_file_folder:
* <a href="https://drive.google.com/drive/folders/1Ka6_iUk7j3ZVnaO3YTNzMXp1D0Mq2im4?usp=share_link">CodeLlama (Transformer-based Retriever)</a>

* <a href="https://drive.google.com/drive/folders/1XRcFTXKU82jkNo50-xjsdvodvLMzWr8t?usp=share_link">CodeLlama (BM25-Retriever)</a>

* <a href="https://drive.google.com/drive/folders/1Ep8F1xEMl1I-8GtNzpJnZa3YgGmuCIsx?usp=share_link">GPT4-o (Transformer-based Retriever)</a>

* <a href="https://drive.google.com/drive/folders/1Ddd5tXehfGt-2eqNm5zkVmomU1GkJSyB?usp=share_link">GPT4-o (BM25-Retriever)</a>

#### Additional Results:  :open_file_folder:

* Wilcoxon signed-rank test comparing CodeLlama and GPT4-o performance across inconsistency levels:

  * <a href="https://drive.google.com/drive/folders/1BDc---DgagM3Ej_w_6FsCI5CX7dIlWwo?usp=sharing"> Transformer-based Retriever
  * <a href="https://drive.google.com/drive/folders/1x_pSlSnZIx-t1HEtRyXz7dft6se0gGvI?usp=sharing">BM25-Retriever</a>      




