import os, argparse, re, json, torch, transformers, sys
import pandas as pd
from tqdm import tqdm
from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever,  BM25Retriever
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
from colorama import Fore, Style
from progress.bar import Bar
from  utilityFunctions import *
from huggingface_hub import login
import evaluate
import logging

# Configure logging
# This will log messages with level INFO or higher to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='inference-adversarial-full-CodeLlama7B-bm25.log', filemode='w')


##################### Supported LLMs #####################               
#                   CodeLlama-{7b,13b}					 #
##########################################################               

DEVICE="cuda"
CONTEXT_MAX_LEN=4096

 
def performQuery(query, dataSet, retriever, args, modelName, model, tokenizer, token):

	def _extractRelevantText(text):
		try:
			match = re.findall(r"Summary: (.+)", text)
			filteredResult = match[1].splitlines()[0].strip()
			return filteredResult
		except Exception:
			return "<NONE>"
	
	result = retriever.retrieve(query=query, top_k=1)
	print(result[0])
	# retrieve the closest document found within the knowledge-base
	data = dataSet[dataSet["Id"] == result[0].meta["id"]]
	retrievedSummary = str(data["comment"].values[0]).strip()
	retrievedMethod = str(data["method"].values[0]).strip()
		
	tr_text = f"Code: {retrievedMethod}\nSummary: {retrievedSummary}\nCode: {query.strip()}\nSummary:"""
	tokensCount = len(tokenizer.encode(tr_text))
	logging.info(f'Prompt Len: {tokensCount}')
	
	shrinkedRetrievedMethod = retrievedMethod
	
	while tokensCount > CONTEXT_MAX_LEN:
		
		methodTokens = shrinkedRetrievedMethod.split()
		methodTokens.pop()
		shrinkedRetrievedMethod = ' '.join(methodTokens)
		tr_text = f"Code: {shrinkedRetrievedMethod}\nSummary: {retrievedSummary}\nCode: {query.strip()}\nSummary:"""
				
		tokensCount = len(tokenizer.tokenize(tr_text))
		logging.info(f"{Fore.RED}Shrinked method: {shrinkedRetrievedMethod} {Style.RESET_ALL}")

	
		
	sequences = model(
		tr_text,
		max_new_tokens=100,
		do_sample=args.do_sample,
					temperature=args.temperature,
		pad_token_id = 50256,
		num_return_sequences=args.num_seq,
		eos_token_id=tokenizer.eos_token_id
	)

	predictions=[]
	for seq in sequences:
		filteredResult = _extractRelevantText(seq['generated_text'])
		predictions.append(filteredResult)
		print(f"{Fore.LIGHTBLUE_EX}{filteredResult}{Style.RESET_ALL}")
		
	return predictions, shrinkedRetrievedMethod, retrievedSummary, tr_text
			


#################################################################
def _getTokenizer(modelName, token=None):
	return AutoTokenizer.from_pretrained(modelName, token=token)


def _getPipeline(modelName,tokenizer,token=None):
	model = transformers.pipeline(
		"text-generation",
		model=modelName,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		token = token,
		tokenizer = tokenizer
	)
	return model
#################################################################
			

def main():
	
	parser = argparse.ArgumentParser(description="Model script")
	parser.add_argument("-m", type= str, help= "[CodeLlama-{7b,13b}, Llama2-{7b,13b}")
	parser.add_argument("-f", type=str, help="CSV File Name")
	parser.add_argument("-do_sample", action="store_true", help="Sampling during decoding")
	parser.add_argument("-num_seq", type=int, default=1, help="The number of independently computed sequences")
	parser.add_argument("-temperature",type=float, default=1.0, help=" The value used to modulate the next token probabilities")
	args = parser.parse_args()

	############################### MODEL SELECTION ##################################
	if args.m == "CodeLlama-7b":
		token = "" #add token
		modelName = "codellama/CodeLlama-7b-hf"
		tokenizer = _getTokenizer(modelName, token)
		model = _getPipeline(modelName,tokenizer,token)


	elif args.m == "CodeLlama-13b":
		token = "" #add token
		modelName = "codellama/CodeLlama-13b-hf"
		tokenizer = _getTokenizer(modelName, token)
		model = _getPipeline(modelName,tokenizer,token)


	else:
		print("Model not supported")
		sys.exit(-1)

	##################################################################################

	
	dataSet = pd.read_csv(args.f, sep=",", engine='python')
	store = ElasticsearchDocumentStore(return_embedding=True)
	
	checkpoint = "sentence-transformers/all-mpnet-base-v2"
	retriever = EmbeddingRetriever(
			document_store=store,
			embedding_model=checkpoint,
			model_format="sentence_transformers",
			max_seq_len=512,
			use_gpu=True
	)
	#retriever = BM25Retriever(store)


	testSet = pd.read_csv(f'../Data/test.csv')
	
	predictions = []
	prompts=[]
	retrievedMethods = []
	retrievedSummaries = []

	for idx,row in testSet.iterrows():
		
		logging.info(f'**********************************************')
		logging.info(f'[+] Instance: {idx}')
		

		pred, retrievedMethod, retrievedSummary, prompt = performQuery(row['method'], dataSet, retriever, args, modelName, model, tokenizer, token)
		
		predictions.append(pred)
		prompts.append(prompt)
		retrievedMethods.append(retrievedMethod)
		retrievedSummaries.append(retrievedSummary)
		logging.info(f'[+]{pred}')
		logging.info(f'[+]{retrievedMethod}')
		logging.info(f'[+]{retrievedSummary}')
		logging.info(f'**********************************************\n')
		

	testSet[f'predictions@{args.num_seq}'] = predictions
	testSet['retrievedMethod'] = retrievedMethods
	testSet['retrievedSummary'] = retrievedSummaries
	testSet['prompt'] = prompts
		
	testSet.to_csv(f'../Data/Experiments/Results/tb-retriever/CodeLlama7B/full-inconsistency.csv')

if __name__ == '__main__':
	main()






