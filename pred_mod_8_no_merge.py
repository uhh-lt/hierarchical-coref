
# predict.py for a 4-way split of the doc along with merging

import argparse

import jsonlines
import torch
from tqdm import tqdm

from coref.coref_model2 import CorefModel
from coref.tokenizer_customization import *
from coref import bert, conll, utils


# usage : python pred_mod_8_no_merge.py roberta data_8_splits_jsonlines/english_test_head.jsonlines output.jsonlines
# output.jsonlines [output path] redundant right now
# pred.conll and gold.conll files written in the data/conll_logs dir, model wts loaded from data/
# the unsplitted doc .jsonlines should be in the data/ dir



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
    doc['cluster_emb'] = [] 
    doc["span_clusters_res"] = []
    doc["words_emb"] = []
    return doc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiment")
    argparser.add_argument("input_file")
    argparser.add_argument("output_file")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in the latest"
                                " weights of the experiment will be loaded;"
                                " if there aren't any, an error is raised.")
    args = argparser.parse_args()

    model = CorefModel(args.config_file, args.experiment)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    model.load_weights(path=args.weights, map_location="cpu",
                       ignore={"bert_optimizer", "general_optimizer",
                               "bert_scheduler", "general_scheduler"})
    model.training = False

    with jsonlines.open(args.input_file, mode="r") as input_data:
        docs = [build_doc(doc, model) for doc in input_data]



    # First run
    with torch.no_grad():
        for doc in tqdm(docs, unit="docs"):
            result, word_emb = model.run(doc)
            print(doc['document_id'])
            doc["words_emb"] = word_emb
            
            doc["span_clusters_res"] = result.span_clusters # predicted span clusters
            doc["word_clusters"] = result.word_clusters
            clusters = doc["span_clusters_res"]
            

            for cluster in clusters:
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

            
            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc[key]

    

    with torch.no_grad():
        docs_new = {} #mapping for doc name to span clusters obtained after merging
        for doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8 in zip(
        docs[::8], docs[1::8], docs[2::8], docs[3::8],
        docs[4::8], docs[5::8], docs[6::8], docs[7::8]
    ):
            
            
            
            clusters1 = doc1["span_clusters_res"]
            
            clusters2 = doc2["span_clusters_res"]
            
            clusters3 = doc3["span_clusters_res"]
            
            clusters4 = doc4["span_clusters_res"]
            
            clusters5 = doc5["span_clusters_res"]
            
            clusters6 = doc6["span_clusters_res"]
            
            clusters7 = doc7["span_clusters_res"]
            
            clusters8 = doc8["span_clusters_res"]
            

            offset1 = 0
            offset2 = len(doc1['cased_words'])
            offset3 = offset2 + len(doc2['cased_words'])
            offset4 = offset3 + len(doc3['cased_words'])
            offset5 = offset4 + len(doc4["cased_words"])
            offset6 = offset5 + len(doc5["cased_words"])
            offset7 = offset6 + len(doc6["cased_words"])
            offset8 = offset7 + len(doc7["cased_words"])


            clusters2 = [[(start + offset2, end + offset2) for start, end in tuple_list] for tuple_list in clusters2]
            clusters3 = [[(start + offset3, end + offset3) for start, end in tuple_list] for tuple_list in clusters3]
            clusters4 = [[(start + offset4, end + offset4) for start, end in tuple_list] for tuple_list in clusters4]
            clusters5 = [[(start + offset5, end + offset5) for start, end in tuple_list] for tuple_list in clusters5]
            clusters6 = [[(start + offset6, end + offset6) for start, end in tuple_list] for tuple_list in clusters6]
            clusters7 = [[(start + offset7, end + offset7) for start, end in tuple_list] for tuple_list in clusters7]
            clusters8 = [[(start + offset8, end + offset8) for start, end in tuple_list] for tuple_list in clusters8]


            
            combined_span_clusters = sorted(clusters1 + clusters2 + clusters3 + clusters4 + clusters5 + clusters6 + clusters7 + clusters8) 

            doc_id = doc1["document_id"][:-2]
            docs_new[doc_id] = combined_span_clusters

    
    data_split = 'test'
    docs = model._get_docs(model.config.__dict__[f"{data_split}_data"])  

    with conll.open_(model.config, model.epochs_trained, data_split) \
            as (gold_f, pred_f):
        pbar = tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            
            doc_id = doc['document_id']
            pred_span_clusters = docs_new[doc_id]

            conll.write_conll(doc, doc["span_clusters"], gold_f)
            # remove singletons using ./coref-toolkit mod --strip-singletons data/conll_logs/roberta_test_e30.gold.conll > data/conll_logs/roberta_test_e30_x.gold.conll
            # then rename it back
            conll.write_conll(doc, pred_span_clusters, pred_f) # will be written in data/conll_logs/ dir
            # to eval : python calculate_conll.py roberta test 30[no of epochs]