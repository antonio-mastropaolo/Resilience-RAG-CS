from haystack.document_stores import ElasticsearchDocumentStore
from haystack import Document
import re, argparse, sys, os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from haystack.nodes import EmbeddingRetriever, BM25Retriever

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE='cuda'

def main():

	parser = argparse.ArgumentParser(description="ElastichSearch doc loader")
	parser.add_argument("-f", type=str, help="CSV File Name")
	parser.add_argument("-t", action= "store_true", help="Whether TSV")
	args = parser.parse_args()


	if args.f == None:
		print("Can't be NULL!")
		sys.exit(-1)
		
		
	if not args.t:
		sep = ","
	else: 
		sep = "\t"

	dataSet = pd.read_csv(args.f, sep=sep, engine='python')
	documents = []
	
	
	for idx, row in dataSet.iterrows():
		#print(row)
		itemToBeStored = row['method'].strip()
		doc = Document(content=itemToBeStored, meta = {"id" : row["Id"], "comment" : row['comment']})
		documents.append(doc)

	store = ElasticsearchDocumentStore(return_embedding=True, recreate_index=True)
	store.delete_labels()
	store.delete_documents()
	store.write_documents(documents = documents)

	#Loading the retrieval model: 
 
	#####################################################################################################
	#We use two different Retrieval Model, one based on BM25 and the other based on Sentence Transformers
 	#####################################################################################################

 
	checkpoint = "sentence-transformers/all-mpnet-base-v2"
	retriever = EmbeddingRetriever(
			document_store=store,
			embedding_model=checkpoint, 
			model_format="sentence_transformers",
   			max_seq_len=512,
			use_gpu=True
	)
 
	store.update_embeddings(retriever = retriever)
	print("[+] Completed[+]\n\n")

if __name__ == "__main__":
	main()
