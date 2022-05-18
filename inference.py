import argparse
import os
import json
import pickle
from src.biosyn import (
    DictionaryDataset,
    BioSyn,
    TextPreprocess
)
from tqdm import tqdm


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='BioSyn Inference')

    # Required
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-subset', required=True, default='test')
    parser.add_argument('-model_name_or_path', required=True, help='Directory for model')
    parser.add_argument('--partition', default='medic')

    # Settings
    parser.add_argument('--show_embeddings',  action="store_true")
    parser.add_argument('--show_predictions',  action="store_true")
    parser.add_argument('--dictionary_path', type=str, default=None, help='dictionary path')
    parser.add_argument('--use_cuda',  action="store_true")
    
    
    args = parser.parse_args()
    return args
    

def cache_or_load_dictionary(biosyn, model_name_or_path, dictionary_path):
    dictionary_name = os.path.splitext(os.path.basename(args.dictionary_path))[0]
    
    cached_dictionary_path = os.path.join(
        './tmp',
        f"cached_{model_name_or_path.split('/')[-1]}_{dictionary_name}.pk"
    )

    # If exist, load the cached dictionary
    if os.path.exists(cached_dictionary_path):
        with open(cached_dictionary_path, 'rb') as fin:
            cached_dictionary = pickle.load(fin)

        dictionary, dict_sparse_embeds, dict_dense_embeds = (
            cached_dictionary['dictionary'],
            cached_dictionary['dict_sparse_embeds'],
            cached_dictionary['dict_dense_embeds'],
        )

    else:
        dictionary = DictionaryDataset(dictionary_path = dictionary_path).data
        dictionary_names = dictionary[:,0]
        dict_sparse_embeds = biosyn.embed_sparse(names=dictionary_names, show_progress=True)
        dict_dense_embeds = biosyn.embed_dense(names=dictionary_names, show_progress=True)
        cached_dictionary = {
            'dictionary': dictionary,
            'dict_sparse_embeds' : dict_sparse_embeds,
            'dict_dense_embeds' : dict_dense_embeds
        }

        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')
        with open(cached_dictionary_path, 'wb') as fin:
            pickle.dump(cached_dictionary, fin)
        print("Saving dictionary into cached file {}".format(cached_dictionary_path))

    return dictionary, dict_sparse_embeds, dict_dense_embeds


def main(mention, args):
    # load biosyn model
    biosyn = BioSyn(
        max_length=25,
        use_cuda=args.use_cuda
    )
    
    biosyn.load_model(model_name_or_path=args.model_name_or_path)

    # preprocess mention
    mention = TextPreprocess().run(mention)

    # embed mention
    mention_sparse_embeds = biosyn.embed_sparse(names=[mention])
    mention_dense_embeds = biosyn.embed_dense(names=[mention])
    
    output = {
        'mention': mention,
    }
    
    if args.show_embeddings:
        output = {
            'mention': mention,
            'mention_sparse_embeds': mention_sparse_embeds.squeeze(0),
            'mention_dense_embeds': mention_dense_embeds.squeeze(0)
        }

    if args.show_predictions:
        if args.dictionary_path == None:
            print('insert the dictionary path')
            return

        # cache or load dictionary
        dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(biosyn, args.model_name_or_path, args.dictionary_path)

        # calcuate score matrix and get top 5
        sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds=mention_sparse_embeds,
            dict_embeds=dict_sparse_embeds
        )
        dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=mention_dense_embeds,
            dict_embeds=dict_dense_embeds
        )
        sparse_weight = biosyn.get_sparse_weight().item()
        hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
        hybrid_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix = hybrid_score_matrix, 
            topk = 10
        )

        # get predictions from dictionary
        predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
        output['predictions'] = []
        
        for prediction in predictions:
            predicted_name = prediction[0]
            predicted_id = prediction[1]
            output['predictions'].append({
                'name': predicted_name,
                'id': predicted_id
            })

        return output['predictions']


def import_pubtator_to_dict(test_filepath):

    input_dictionary = {}
    gold_standard = {}

    with open(test_filepath, 'r') as in_file:
        data = in_file.readlines()
        in_file.close()

    for line in data:
        line_data = line.split('\t')
        
        if len(line_data) == 6:    
            doc_id = line_data[0]
            mention = line_data[3]
            true_kb_id = line_data[5].strip('\n')

            if doc_id in input_dictionary.keys():
                input_dictionary[doc_id].append(mention)
                gold_standard[doc_id][mention] = true_kb_id
            
            else:
                input_dictionary[doc_id] = [mention]
                gold_standard[doc_id] = {mention: true_kb_id}

    return input_dictionary, gold_standard


def calculate_metrics(tp,fp,fn):

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = tp / (tp+fp)

    return accuracy, precision, recall, f1_score 


def filter_top_pred(predictions, true_kb_id):
    """If the top prediction is associated with the true_kb_id, the top 
    prediction is replaced by the first prediction that is not associated with 
    the true_kb_id"""

    prediction_found = False
    top_prediction = None

    for pred in predictions:

        if pred['id'] != true_kb_id:
            prediction_found = True
            top_prediction = pred['id']
        
        if prediction_found:
            break

    return top_prediction
    

def import_input(filepath):
    """Import mentions from json file into dict"""

    gold_standard = {}

    with open(filepath, 'r') as in_file:
        gold_standard = json.load(in_file)
        in_file.close()

    return gold_standard


if __name__ == '__main__':

    args = parse_args()

    # Import mentions of the test set
    test_filepath = ''

    if args.dataset == 'evanil':
        
        if args.subset=='test_refined':
            test_filepath = 'datasets/evanil/{}/test_refined.json'.format(args.partition)

        elif args.subset =='test':
            test_filepath = 'datasets/evanil/{}/test.json'.format(args.partition)

    elif args.dataset != 'evanil':
        
        if args.subset=='test_refined':
            test_filepath = '../data/corpora/preprocessed/{}_WORKS/test_refined.json'.\
                format(args.dataset)
        
        elif args.subset =='test':
            test_filepath = '../data/corpora/preprocessed/{}_WORKS/test.json'.\
                format(args.dataset)
    
    gold_standard = import_input(test_filepath)
    
    # Iterate over each mention and apply model
    tp = 0
    fp = 0
    fn = 0
    nil_count = 0
    total_count = 0

    pbar = tqdm(total=len(gold_standard.keys()))

    for doc in gold_standard:
        mentions = gold_standard[doc]
        
        for mention in mentions: 
            total_count += 1
    
            if args.dataset == 'evanil':
                ancestor_id = mentions[mention][1]
                true_kb_id = mentions[mention][0]
                predictions = main(mention, args)
                
                top_pred_id = filter_top_pred(predictions, true_kb_id)
                
                if top_pred_id == ancestor_id:
                    tp += 1

                elif top_pred_id != ancestor_id:

                    if top_pred_id == None or top_pred_id == '':
                        fn += 1
                    
                    else:
                        fp += 1 
            
            else:
                mention_text = mention[1]
                correct_answer = mention[0]
                predictions = main(mention_text, args)
                top_pred = predictions[0]['id']
                answers = []
                answers_tmp= []

                if '|' in top_pred:
                    answers_tmp = top_pred.split('|')
                else:
                    answers_tmp = [top_pred]

                for kb_id in answers_tmp:
                    
                    if kb_id[0] == 'D' or kb_id[0] == 'C':
                        kb_id = 'MESH_' + kb_id
                    
                    elif kb_id[0] == 'M':
                        continue

                    else:
                        kb_id = 'OMIM_' + kb_id
                    
                    answers.append(kb_id)
               
                found_match = False

                if correct_answer == 'MESH_-1':
                    nil_count += 1
                
                elif correct_answer == '-1':
                    nil_count += 1
                        
                else:
                       
                    for kb_id in answers:
                        
                        if not found_match:
                            
                            if kb_id in correct_answer:
                                found_match = True     

                if found_match:
                    tp += 1
        
                else:
                    fp += 1
                       
        pbar.update(1)
        
    pbar.close()

    #--------------------------------------------------------------------------
    #                       CALCULATE METRICS
    #--------------------------------------------------------------------------
    
    if args.dataset == 'evanil':
        print("tp:", tp, "\nfp:", fp, "\nfn:", fn)
        accuracy, precision, recall, f1_score = calculate_metrics(tp,fp,fn)
        
        print("Results\nPrecision: {}\nRecall: {}\nF1-score: {}\nAccuracy: {}".format(
            precision, recall, f1_score, accuracy))

    elif args.dataset != 'evanil':
        doc_count = int(len(gold_standard.keys()))
    
        accuracy = tp/(tp + fp)
    
        stats = ""
        stats += "------\nNumber of documents: " + str(doc_count)  
        stats += "\nTotal entities (NIL+non-NIL): " + str(total_count)
        stats += "\nTrue NIL entities: " + str(nil_count)
        stats += "\nTrue Positives: " + str(tp)
        stats += "\nFalse Positives: " + str(fp)
        stats += "\nAccuracy: " + str(accuracy)

        print(stats)