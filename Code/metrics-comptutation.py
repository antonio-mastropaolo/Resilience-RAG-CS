import pandas as pd
import evaluate
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import os
import torch.nn.functional as F
from sentence_transformers import evaluation

DEVICE = "cuda:0"

#########################################

sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")

########################################


################# SIDE #################


# specify the path to the best-performing checkpoint
checkPointFolder = "./cs-metric/models/triplet-loss/hard_negatives/141205"
tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
sideModel = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


######################################


def levenshtein_normalized(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return matrix[size_x - 1, size_y - 1]/max(size_x, size_y)


def performEvalMetricsComputation(prediction, ground_truth, java_method):

    ############ SIDE COMPUTATION #############
    pair = [java_method, prediction]

    encoded_input = tokenizer(
        pair, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Compute token embeddings
    with torch.no_grad():
        model_output = sideModel(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    sideScore = util.pytorch_cos_sim(
        sentence_embeddings[0], sentence_embeddings[1]).item()

    sideScore = round(sideScore, 2)

    sacrebleuComputation = sacrebleu.compute(
        predictions=[prediction], references=[[ground_truth]])
    sacrebleuScore = round(sacrebleuComputation["score"], 2)

    meteorComputation = meteor.compute(
        predictions=[prediction], references=[ground_truth])
    meteorScore = round(meteorComputation["meteor"], 2) * 100

    rougeComputation = rouge.compute(
        predictions=[prediction], references=[ground_truth])
    rougeScore = round(rougeComputation["rougeL"], 2) * 100

    chrFComputation = chrf.compute(
        predictions=[prediction], references=[ground_truth])
    chrFScore = round(chrFComputation['score'], 2)

    levDistance = levenshtein_normalized(prediction.split(), ground_truth.split())

    return sacrebleuScore, meteorScore, rougeScore, chrFScore, levDistance, sideScore


def main():

    df_baseline = pd.read_csv(
        '../Data/Experiments/Results/tb-retriever/CodeLlama7B/baseline.csv')
    df_inc_10 = pd.read_csv(
        '../Data/Experiments/Results/tb-retriever/CodeLlama7B/10-inconsistency.csv')
    df_inc_20 = pd.read_csv(
        '../Data/Experiments/Results/tb-retriever/CodeLlama7B/20-inconsistency.csv')
    df_inc_full = pd.read_csv(
        '../Data/Experiments/Results/tb-retriever/CodeLlama7B/full-inconsistency.csv')

    methods = [item.strip() for item in list(df_baseline['method'])]
    gt_summaries = [item.strip().lower() for item in list(df_baseline['comment'])]
    
    predictionsBaseline = [eval(row["predictions@1"])[0].strip().lower() for _, row in df_baseline.iterrows()]
    predictionsIncFull = [eval(row["predictions@1"])[0].strip().lower() for _, row in df_inc_full.iterrows()]
    predictionsInc10 = [eval(row["predictions@1"])[0].strip().lower() for _, row in df_inc_10.iterrows()]
    predictionsInc20 = [eval(row["predictions@1"])[0].strip().lower() for _, row in df_inc_20.iterrows()]

    # retSummaryBaseline = [item.strip().lower()
    #                       for item in list(df_baseline['retrievedSummary'])]
    # retSummaryIncFull = [item.strip().lower()
    #                      for item in list(df_inc_full['retrievedSummary'])]
    # retSummaryInc10 = [item.strip().lower()
    #                    for item in list(df_inc_10['retrievedSummary'])]
    # retSummaryInc20 = [item.strip().lower()
    #                    for item in list(df_inc_20['retrievedSummary'])]

    sacreBleuBaselineList, meteorBaselineList, rougeBaselineList, levDistanceBaselineList, chrfBaselineList, sideBaselineList = [], [], [], [], [], []
    sacreBleuIncFullList, meteorIncFullList, rougeIncFullList, levDistanceIncFullList, chrfIncFullList, sideIncFullList = [], [], [], [], [], []
    sacreBleuInc10List, meteorInc10List, rougeInc10List, levDistanceInc10List, chrfInc10List, sideInc10List = [], [], [], [], [], []
    sacreBleuInc20List, meteorInc20List, rougeInc20List, levDistanceInc20List, chrfInc20List, sideInc20List = [], [], [], [], [], []

    for method, predictionBaseline, predictionIncFull, predictionInc10, predictionInc20, comment in tqdm(zip(methods, predictionsBaseline, predictionsIncFull, predictionsInc10, predictionsInc20, gt_summaries)):

        sacrebleuScoreBaseline, meteorScoreBaseline, rougeScoreBaseline, chrfScoreBaseline, levBaseline, sideBaseline = performEvalMetricsComputation(
            predictionBaseline, comment, method)

        sacreBleuBaselineList.append(sacrebleuScoreBaseline)
        meteorBaselineList.append(meteorScoreBaseline)
        rougeBaselineList.append(rougeScoreBaseline)
        sideBaselineList.append(sideBaseline)
        chrfBaselineList.append(chrfScoreBaseline)
        levDistanceBaselineList.append(levBaseline)

        sacrebleuScoreIncFull, meteorScoreIncFull, rougeScoreIncFull, chrfScoreIncFull, levIncFull, sideIncFull = performEvalMetricsComputation(
            predictionIncFull, comment, method)
        
        sacreBleuIncFullList.append(sacrebleuScoreIncFull)
        meteorIncFullList.append(meteorScoreIncFull)
        rougeIncFullList.append(rougeScoreIncFull)
        chrfIncFullList.append(chrfScoreIncFull)
        levDistanceIncFullList.append(levIncFull)
        sideIncFullList.append(sideIncFull)

        sacrebleuScoreInc10, meteorScoreInc10, rougeScoreInc10,  chrfScoreInc10, levInc10, sideInc10 = performEvalMetricsComputation(
            predictionInc10, comment, method)

        sacreBleuInc10List.append(sacrebleuScoreInc10)
        meteorInc10List.append(meteorScoreInc10)
        rougeInc10List.append(rougeScoreInc10)
        chrfInc10List.append(chrfScoreInc10)
        levDistanceInc10List.append(levInc10)
        sideInc10List.append(sideInc10)

        sacrebleuScoreInc20, meteorScoreInc20, rougeScoreInc20, chrfScoreInc20, levInc20, sideInc20 = performEvalMetricsComputation(
            predictionInc20, comment, method)

        sacreBleuInc20List.append(sacrebleuScoreInc20)
        meteorInc20List.append(meteorScoreInc20)
        rougeInc20List.append(rougeScoreInc20)
        chrfInc20List.append(chrfScoreInc20)
        levDistanceInc20List.append(levInc20)
        sideInc20List.append(sideInc20)

    #################################################################
    df_baseline['sacreBleu'] = sacreBleuBaselineList
    df_baseline['Meteor'] = meteorBaselineList
    df_baseline['ROUGE-LCS'] = rougeBaselineList
    df_baseline['Levenshtein-Distance'] = levDistanceBaselineList
    df_baseline['chrF'] = chrfBaselineList
    df_baseline['Side'] = sideBaselineList
    #################################################################

    #################################################################
    df_inc_full['sacreBleu'] = sacreBleuIncFullList
    df_inc_full['Meteor'] = meteorIncFullList
    df_inc_full['ROUGE-LCS'] = rougeIncFullList
    df_inc_full['Levenshtein-Distance'] = levDistanceIncFullList
    df_inc_full['chrF'] = chrfIncFullList
    df_inc_full['Side'] = sideIncFullList
    #################################################################

    #################################################################
    df_inc_10['sacreBleu'] = sacreBleuInc10List
    df_inc_10['Meteor'] = meteorInc10List
    df_inc_10['ROUGE-LCS'] = rougeInc10List
    df_inc_10['Levenshtein-Distance'] = levDistanceInc10List
    df_inc_10['chrF'] = chrfInc10List
    df_inc_10['Side'] = sideInc10List
    #################################################################

    #################################################################
    df_inc_20['sacreBleu'] = sacreBleuInc20List
    df_inc_20['Meteor'] = meteorInc20List
    df_inc_20['ROUGE-LCS'] = rougeInc20List
    df_inc_20['Levenshtein-Distance'] = levDistanceInc20List
    df_inc_20['chrF'] = chrfInc20List
    df_inc_20['Side'] = sideInc20List
    #################################################################

    df_baseline.to_csv(
        '../Data/Experiments/Results/tb-Retriever/CodeLlama7B/baseline-results.csv', index=False)
    df_inc_10.to_csv(
        '../Data/Experiments/Results/tb-Retriever/CodeLlama7B/10-inconsistency-results.csv', index=False)
    df_inc_20.to_csv(
        '../Data/Experiments/Results/tb-Retriever/CodeLlama7B/20-inconsistency-results.csv', index=False)
    df_inc_full.to_csv(
        '../Data/Experiments/Results/tb-Retriever/CodeLlama7B/full-inconsistency-results.csv', index=False)

    print("Completed!\n\n")


if __name__ == "__main__":
    main()
