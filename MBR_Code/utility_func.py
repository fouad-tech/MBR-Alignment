# TODO: Put all the utility functions here.
# import re
import numpy as np
from nltk.tokenize import ToktokTokenizer
import ast
from evaluate import load



def load_similarity(sim):

    if sim == "bertscore":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            return similarity.compute(predictions=hyp, references=ref, lang="en")["f1"]

    elif sim == "sacrebleu":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            scores = [
                similarity.compute(predictions=[hyp[i]], references=[ref[i]])["score"]
                for i in range(len(hyp))
            ]
            return scores

    elif sim == "unigramf1":
        similarity = ToktokTokenizer()

        def compute_similarity(hyp, ref, src):
            nhyp = len(hyp)
            f1s = []
            for i in range(nhyp):
                h = hyp[i]
                r = ref[i]
                hyp_tok = similarity.tokenize(h)
                ref_tok = similarity.tokenize(r)

                if len(hyp_tok) == 0 or len(ref_tok) == 0:
                    f1s.append(0.0)
                else:
                    precision = len(
                        [token for token in hyp_tok if token in ref_tok]
                    ) / len(hyp_tok)
                    recall = len(
                        [token for token in hyp_tok if token in ref_tok]
                    ) / len(ref_tok)

                    if precision + recall < 0.0001:
                        # Prevent zero division.
                        f1s.append(0.0)
                    else:
                        f1s.append(2.0 * precision * recall / (precision + recall))
            return f1s

    else:
        assert False

    return compute_similarity, similarity


def load_distance(sim, compute_similarity):
    if sim != "sacrebleu":

        def compute_distance(hyp, ref, src):
            return [1.0 - sim for sim in compute_similarity(hyp, ref, src)]

    else:
        # sacrebleu ranges (0, 100), so need to normalize it.
        def compute_distance(hyp, ref, src):
            return [1.0 - sim / 100.0 for sim in compute_similarity(hyp, ref, src)]

    return compute_distance


def load_evaluate(eval_func, sim, similarity):
    if eval_func == "bleurt":
        evaluator = load(eval_func, checkpoint="BLEURT-LARGE-512")
    elif eval_func =='rouge1' or eval_func== 'rouge2':
        evaluator = load('rouge')
    elif eval_func == 'bleu_bp' or eval_func == 'bleu_lr':
        evaluator = load('sacrebleu')
    else:
        evaluator = load(eval_func)

    if eval_func == "rouge":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]
    
    elif eval_func == "rouge1":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rouge1"]
    
    elif eval_func == "rouge2":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rouge2"]
    
    elif eval_func == "bleurt":

        #def compute_evaluate(hyp, ref, src):
        #    return evaluator.compute(predictions=[hyp], references=[ref])["scores"][0]
    
        def compute_evaluate(hyp, ref, src):
            scores = [
                evaluator.compute(predictions=[hyp], references=[r])["scores"][0]
                for r in ref
            ]
            return max(scores)

    elif eval_func == "meteor":
        #def compute_evaluate(hyp, ref, src):
        
        #    scores = [
        #        evaluator.compute(predictions=[hyp], references=[r])["meteor"]
        #        for r in ref
        #    ]
        #    return max(scores)

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["meteor"]

    elif eval_func == "sacrebleu":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["score"]
    
    elif eval_func == "bleu_bp":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["bp"]

    elif eval_func == "bleu_lr":

        def compute_evaluate(hyp, ref, src):
            ev = evaluator.compute(predictions=[hyp], references=[ref])
            return ev['sys_len']/ev['ref_len']

    elif eval_func == "sacrebleuzh":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(
                predictions=[hyp], references=[ref], tokenize="zh"
            )["score"]

    elif eval_func == 'bertscore':
        #def compute_evaluate(hyp, ref, src):
        # scores = []
        # ref = ast.literal_eval(ref)
        # for r in ref:
   
        #    x = evaluator.compute(predictions=[hyp], references=[r], lang="en")["f1"][0]
        #    scores.append(x)
        # return max(scores)
        
        #def compute_evaluate(hyp, ref, src):
         
        #    scores = [
        #        evaluator.compute(predictions=[hyp], references=[r], lang="en")["f1"][0]
        #        for r in ref
        #    ]
        #    return max(scores)

        def compute_evaluate(hyp, ref, src):
            x = evaluator.compute(predictions=[hyp], references=[ref], lang="en")["f1"]
            
            return x[0]

    else:
        assert False

    return compute_evaluate, evaluator


def compute_self_score(hyps, src, compute_evaluate):
    scores = []
    n_samples = 0
    n = len(hyps)
    for i in range(n):
        for j in range(n):
            if i != j:
                score = compute_evaluate(hyps[i], hyps[j], src)
                scores.append(score)
                n_samples += 1
    return sum(scores) / n_samples
