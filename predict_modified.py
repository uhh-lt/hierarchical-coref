import argparse
import itertools

import jsonlines
import torch
from tqdm import tqdm

import conll_eval
from coref.coref_model2 import CorefModel as CorefModel
from coref.tokenizer_customization import *
from coref import bert, conll, utils


# usage : python predict_modified.py roberta litbank_splitted/jsonlines/english_test_head.jsonlines
# pred.conll and gold.conll files written in the data/conll_logs dir, model wts loaded from data/
# the unsplitted doc .jsonlines should be in the data/ dir

def build_cluster_emb(doc1, doc2, clusters, offset):
    # offset is 0 for doc1, doc2
    # offset is len*word1 + lenword2
    
    words_emb1 = doc1["words_emb"]  
    words_emb2 = doc2["words_emb"] 
    word_emb = torch.cat((words_emb1, words_emb2), 0)
    # task : see the wordsemb1 type

    cluster_emb = []
    for cluster in clusters:
        cluster_i = []
        for span in cluster:
            
            span_embedding = None
            start, end = span
            start -= offset
            end -= offset

            for i in range(start, end):
                if(span_embedding == None):
                    span_embedding = word_emb[i]
                else:
                    span_embedding += word_emb[i]
            span_embedding /= (end - start)
            cluster_i.append(span_embedding)
            
        cluster_i = torch.stack(cluster_i)
        cluster_i = torch.mean(cluster_i, dim=0)
        cluster_emb.append(cluster_i)

    return cluster_emb, word_emb


def build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(model.config.bert_model,
                                        lambda _: True)
    token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})

    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (token_map[word]
                          if word in token_map
                          else model.tokenizer.tokenize(word))
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id

    doc["head2span"] = []
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []
    doc["cluster_emb"] = [] 
    doc["span_clusters_res"] = []
    return doc


def add_cluster_embeddings_to_docs(model, docs):
    # building the cluster embeddings
    with torch.no_grad():
        for doc in tqdm(docs, unit="docs"):
            result, word_emb = model.run(doc)
            doc['cluster_emb'] = []
            doc["words_emb"] = word_emb
            doc["span_clusters_res"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters
            clusters = doc["span_clusters_res"]
            

            for cluster in clusters:
                # you have to set a offset 
                cluster_i = []
                for span in cluster:
                    
                    span_embedding = None
                    start, end = span
                    for i in range(start, end):
                        if(span_embedding == None):
                            span_embedding = word_emb[i]
                        else:
                            span_embedding += word_emb[i]
                    span_embedding /= (end - start)
                    cluster_i.append(span_embedding)
                    
                cluster_i = torch.stack(cluster_i)
                cluster_i = torch.mean(cluster_i, dim=0)
                doc['cluster_emb'].append(cluster_i)


def merge_two_docs(doc1, doc2, new_name, merge=True):
    doc1 = build_doc(doc1, model)
    doc2 = build_doc(doc2, model)
    add_cluster_embeddings_to_docs(model, [doc1, doc2])
    span_clusters_mapping = {}
    cluster_emb1 = doc1['cluster_emb']
    clusters1 = doc1['span_clusters_res']
    cluster_emb2 = doc2['cluster_emb']
    clusters2 = doc2['span_clusters_res']
    offset = len(doc1['cased_words'])
    
    clusters2 = [[(start + offset, end + offset) for start, end in tuple_list] for tuple_list in clusters2]
    cluster_emb_merged = torch.stack(cluster_emb1 + cluster_emb2)
    cluster_emb_merged = cluster_emb_merged.to('cuda')
    for i, cluster in enumerate(clusters1 + clusters2):
        span_clusters_mapping[i] = cluster #List[Tuple[int, int]]

    combined_span_clusters = []
    if merge: 
        res = model_merging.run2(cluster_emb_merged)
    
        #mapping the indexes to the actual clusters of spans
        for second_lvl_clusters in res.word_clusters:
            combined_span_clusters_i = []
            for x in second_lvl_clusters:
                combined_span_clusters_i += span_clusters_mapping[x]
            combined_span_clusters.append(sorted(combined_span_clusters_i))
    cluster_emb, words_emb = build_cluster_emb(doc1, doc2, clusters1 + clusters2, 0)
    new_doc = {
        "document_id": new_name,
        "words_emb": words_emb,
        "cased_words": doc1["cased_words"] + doc2["cased_words"],
        "sent_id": doc1["sent_id"] + [i + doc1["sent_id"][-1] for i in doc2["sent_id"]],
        "part_id": doc1["part_id"],
        "speaker": doc1["speaker"] + doc2["speaker"],
        "pos": doc1["pos"] + doc2["pos"],
        "deprel": doc1["deprel"] + doc2["deprel"],
        "head": doc1["head"] + [h + len(doc1["head"]) if h is not None else None for h in doc2["head"]],
        "word_clusters": None,
        "span_clusters": None,
        "cluster_emb": cluster_emb,
        "span_clusters_res": combined_span_clusters if merge else clusters1 + clusters2,
    }
    return new_doc


def pairs(iterable, n=2):
    args = [iter(iterable)] * n
    return zip(*args, strict=True)


def merge_matching_names(docs, merge=True):
    docs_new = {} #mapping for doc name to span clusters obtained after merging
    grouped_docs = itertools.groupby(docs, key=lambda d: d["document_id"].rsplit("_", 1)[0])
    grouped_docs = {key: list(value) for key, value in grouped_docs}
    grouped_names = {key: [v["document_id"] for v in value] for key, value in grouped_docs.items()}
    for doc_id, to_merge in grouped_docs.items():
        while len(to_merge) > 1:
            with torch.no_grad():
                to_merge = [merge_two_docs(a, b, doc_id, merge) for a, b in pairs(to_merge)]
        docs_new[doc_id] = to_merge[0]
        print(doc_id, len(to_merge))
    return docs_new


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiment")
    argparser.add_argument("input_file")
    # argparser.add_argument("output_file")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--no-merge", action="store_true")
    argparser.add_argument("--final-docs", default=None, type=str)
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in the latest"
                                " weights of the experiment will be loaded;"
                                " if there aren't any, an error is raised.")
    argparser.add_argument("--weights-merging",
                           help="Path to file with weights to load for merging."
                                " If not supplied, in the latest"
                                " weights of the experiment will be loaded;"
                                " if there aren't any, an error is raised.")
    args = argparser.parse_args()

    model = CorefModel(args.config_file, args.experiment)
    model_merging = CorefModel(args.config_file, args.experiment)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    model.load_weights(path=args.weights, map_location="cpu",
                       ignore={"bert_optimizer", "general_optimizer",
                               "bert_scheduler", "general_scheduler"})
    model.training = False
    model_merging.load_weights(path=args.weights_merging or args.weights, map_location="cpu",
                       ignore={"bert_optimizer", "general_optimizer",
                               "bert_scheduler", "general_scheduler"})
    model_merging.training = False

    with jsonlines.open(args.input_file, mode="r") as input_data:
        docs = [build_doc(doc, model) for doc in input_data]

    add_cluster_embeddings_to_docs(model, docs)
    # with jsonlines.open(args.output_file, mode="w") as output_data:
    #     output_data.write_all(docs_new)

    docs_new = merge_matching_names(docs, not args.no_merge)
    
    data_split = 'test'

    docs_location = args.final_docs or model.config.__dict__[f"{data_split}_data"]
    print("Loading docs from", docs_location)
    docs = model._get_docs(docs_location)   # from the head.jsonlines, because they contain 'span_clusters' not the other .jsonlines which contains the 'clusters'
    # span clusters are formed after you run the convert_to_heads.py -- which are : clusters - some deleted clusters
    
    with conll.open_(model.config, model.epochs_trained, data_split) \
            as (gold_f, pred_f):
        pbar = tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            
            doc_id = doc['document_id']
            pred_span_clusters = docs_new[doc_id]["span_clusters_res"]
            conll.write_conll(doc, doc["span_clusters"], gold_f)
            # remove singletons using ./coref-toolkit mod --strip-singletons data/conll_logs/roberta_test_e30.gold.conll > data/conll_logs/roberta_test_e30_x.gold.conll
            # then rename it back
            conll.write_conll(doc, pred_span_clusters, pred_f) # will be written in data/conll_logs/ dir
            # to eval : python calculate_conll.py roberta test 30[no of epochs]
        names = gold_f.name, pred_f.name
    conll_eval.main(*names)