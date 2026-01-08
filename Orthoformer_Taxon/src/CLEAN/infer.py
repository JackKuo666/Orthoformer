import torch
from src.CLEAN.utils import * 
from src.CLEAN.model import LayerNormNet
from src.CLEAN.distance_map import *
from src.CLEAN.evaluate import *
import pandas as pd
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def infer_pvalue(train_data, test_data, p_value = 1e-5, nk_random = 20, 
                 report_metrics = False, pretrained=True, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
        
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    write_pvalue_choices( eval_df, out_filename, random_nk_dist_map, p_value=p_value)
    # optionally report prediction precision/recall/...
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f'############ EC calling results using random '
        f'chosen {nk_random}k samples ############')
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)  
    




def infer_maxsep(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, gmm = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(384, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    # write_max_sep_choices(eval_df, out_filename, gmm=gmm, distance_threshold=5)
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} '
             f'| acc: {acc:.3}')
        print('-' * 75)

         # Create metrics dictionary
        metrics_dict = {
            'train_data': train_data,
            'test_data': test_data,
            'total_samples': len(true_label),
            'total_ec': len(all_label),
            'precision': pre,
            'recall': rec,
            'f1_score': f1,
            'auc': roc,
            'accuracy': acc,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name if model_name else 'pretrained',
            'gmm_used': gmm is not None
        }
        
        # Save metrics to CSV file
        metrics_filename = f"results/metrics_{train_data}_{test_data}.csv"
        ensure_dirs("./results")
        
        if os.path.exists(metrics_filename):
            existing_df = pd.read_csv(metrics_filename)
            new_df = pd.DataFrame([metrics_dict])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(metrics_filename, index=False)
        else:
            pd.DataFrame([metrics_dict]).to_csv(metrics_filename, index=False)
        
        # Save detailed prediction results
        all_predictions_df = save_detailed_predictions(pred_label, true_label, pred_probs, 
                                                      id_ec_test, train_data, test_data)
        
        # Save exact matches and all correct predictions (exact matches + partial matches) separately
        exact_matches_df, all_correct_df = save_separate_match_files(all_predictions_df, train_data, test_data)
        
        # Also save wrong predictions for analysis
        wrong_predictions_df = all_predictions_df[all_predictions_df['correct'] == False]
        wrong_filename = f"results/wrong_predictions_{train_data}_{test_data}.csv"
        wrong_predictions_df.to_csv(wrong_filename, index=False)
        
        # Count match types
        exact_matches_count = len(exact_matches_df)
        partial_matches_count = len(all_correct_df) - exact_matches_count
        all_correct_count = len(all_correct_df)
        
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)}')
        print(f'>>> precision: {pre:.3} | recall: {rec:.3} | F1: {f1:.3} | AUC: {roc:.3} | Accuracy: {acc:.3}')
        print(f'>>> All correct predictions: {all_correct_count}/{len(true_label)} ({all_correct_count/len(true_label)*100:.1f}%)')
        print(f'>>>   - Exact matches: {exact_matches_count} ({exact_matches_count/len(true_label)*100:.1f}%)')
        print(f'>>>   - Partial matches: {partial_matches_count} ({partial_matches_count/len(true_label)*100:.1f}%)')
        print(f'>>> Wrong predictions: {len(wrong_predictions_df)}/{len(true_label)} ({len(wrong_predictions_df)/len(true_label)*100:.1f}%)')
        print(f'>>> Metrics saved to: {metrics_filename}')
        print(f'>>> Exact matches saved to: results/exact_matches_{train_data}_{test_data}.csv')
        print(f'>>> All correct predictions saved to: results/all_correct_predictions_{train_data}_{test_data}.csv')
        print(f'>>> Wrong predictions saved to: {wrong_filename}')
        print(f'>>> All predictions saved to: results/all_predictions_{train_data}_{test_data}.csv')
        print('-' * 75)
        
        return metrics_dict, exact_matches_df, all_correct_df, wrong_predictions_df
    
    return None

def save_separate_match_files(all_predictions_df, train_data, test_data):
    """
    Save exact matches and all correct predictions to different CSV files
    For partially matched samples, keep all prediction results
    """
    # Extract exact match samples (match_type == 'exact')
    exact_matches_df = all_predictions_df[all_predictions_df['match_type'] == 'exact'].copy()
    exact_filename = f"results/exact_matches_{train_data}_{test_data}.csv"
    exact_matches_df.to_csv(exact_filename, index=False)
    
    # Extract all correct prediction samples (including exact matches and partial matches)
    all_correct_df = all_predictions_df[all_predictions_df['correct'] == True].copy()
    
    # Add matched EC information for partially matched samples
    all_correct_df = add_matched_ec_info(all_correct_df)
    
    all_correct_filename = f"results/all_correct_predictions_{train_data}_{test_data}.csv"
    all_correct_df.to_csv(all_correct_filename, index=False)
    
    print(f"Exact match samples saved to: {exact_filename} ({len(exact_matches_df)} samples)")
    print(f"All correct prediction samples saved to: {all_correct_filename} ({len(all_correct_df)} samples)")
    
    return exact_matches_df, all_correct_df

def add_matched_ec_info(df):
    """
    Add matched EC information for partially matched samples
    """
    result_df = df.copy()
    
    # Add new columns to store matched ECs
    result_df['matched_ec'] = ''
    result_df['all_predicted_ecs'] = ''
    result_df['matching_ecs'] = ''
    
    for idx, row in result_df.iterrows():
        pred_ec = row['predicted_ec']
        true_ec = row['true_ec']
        
        # Parse predicted EC and true EC as lists
        pred_list = parse_ec_string(pred_ec)
        true_list = parse_ec_string(true_ec)
        
        # Store all predicted ECs
        result_df.at[idx, 'all_predicted_ecs'] = ';'.join(pred_list)
        
        # Find matched ECs
        matched_ecs = list(set(pred_list) & set(true_list))
        result_df.at[idx, 'matching_ecs'] = ';'.join(matched_ecs)
        
        # If there are multiple matched ECs, take the first as the primary match
        if matched_ecs:
            result_df.at[idx, 'matched_ec'] = matched_ecs[0]
    
    return result_df

def parse_ec_string(ec_string):
    """
    Parse EC string to list
    Handle various formats: string, list string, etc.
    """
    if pd.isna(ec_string) or ec_string == '':
        return []
    
    # If already a list, return directly
    if isinstance(ec_string, list):
        return [str(ec) for ec in ec_string]
    
    # Convert to string
    ec_str = str(ec_string)
    
    # Handle different separators
    if ';' in ec_str:
        return [ec.strip() for ec in ec_str.split(';') if ec.strip()]
    elif ',' in ec_str:
        return [ec.strip() for ec in ec_str.split(',') if ec.strip()]
    elif ' ' in ec_str and len(ec_str.split()) > 1:
        return [ec.strip() for ec in ec_str.split() if ec.strip()]
    else:
        # Try to parse strings like "['1883', '2995706']"
        if ec_str.startswith('[') and ec_str.endswith(']'):
            try:
                # Use ast for safe parsing
                import ast
                parsed_list = ast.literal_eval(ec_str)
                if isinstance(parsed_list, list):
                    return [str(item) for item in parsed_list]
            except:
                pass
        
        # If cannot parse as list, return single-element list
        return [ec_str]

def save_detailed_predictions(pred_labels, true_labels, pred_probs, id_ec_test, train_data, test_data):
    """
    Save detailed prediction results and handle list-type predictions
    """
    # Get test sample names
    test_sample_names = list(id_ec_test.keys())
    
    # Ensure consistent lengths
    min_length = min(len(test_sample_names), len(pred_labels), len(true_labels))
    test_sample_names = test_sample_names[:min_length]
    pred_labels = pred_labels[:min_length]
    true_labels = true_labels[:min_length]
    
    # Collect all prediction results
    all_predictions_data = []
    
    for i, (sample_name, pred, true) in enumerate(zip(test_sample_names, pred_labels, true_labels)):
        # Check if prediction is correct
        is_correct = is_prediction_correct(pred, true)
        
        # Determine match type
        if is_correct:
            if pred == true:
                match_type = 'exact'
            else:
                match_type = 'partial'
        else:
            match_type = 'none'
        
        prediction_data = {
            'sample_name': sample_name,
            'predicted_ec': str(pred),
            'true_ec': str(true),
            'correct': is_correct,
            'match_type': match_type,  # Ensure match_type is added
            'train_data': train_data,
            'test_data': test_data,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add prediction probability
        if pred_probs is not None and i < len(pred_probs):
            prediction_data['prediction_confidence'] = pred_probs[i]
        
        all_predictions_data.append(prediction_data)
    
    # Create DataFrame
    all_predictions_df = pd.DataFrame(all_predictions_data)
    
    # Save all predictions
    all_filename = f"results/all_predictions_{train_data}_{test_data}.csv"
    all_predictions_df.to_csv(all_filename, index=False)
    
    return all_predictions_df

def is_prediction_correct(pred, true):
    """
    Check if prediction is correct, handling the case where prediction is a list or array.
    For example: pred = ['1883', '2995706'], true = ['1883'] should return True
    """
    # Ensure inputs are in list form if they aren't already
    if not isinstance(pred, (list, tuple, np.ndarray)):
        pred = [pred]
    if not isinstance(true, (list, tuple, np.ndarray)):
        true = [true]
    
    # Convert elements to strings for comparison (ensure consistency)
    pred = [str(p) for p in pred]
    true = [str(t) for t in true]
    
    # Use set intersection to check if any predicted EC matches the true EC
    return len(set(pred) & set(true)) > 0



## functions for inference on the fly ( using input embeddings)

# def infer_maxsep_from_embeddings(train_data, test_data, report_metrics = False, 
#                  pretrained=True, model_name=None, gmm = None):
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     dtype = torch.float32
#     id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
#     id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
#     # load checkpoints
#     # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
#     # and rebuild with [python build.py install]
#     # if inferencing on model trained with supconH loss
#     model = LayerNormNet(384, 128, device, dtype)
    
#     if pretrained:
#         try:
#             checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
#         except FileNotFoundError as error:
#             raise Exception('No pretrained weights for this training data')
#     else:
#         try:
#             checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
#         except FileNotFoundError as error:
#             raise Exception('No model found!')
            
#     model.load_state_dict(checkpoint)
#     model.eval()
#     # load precomputed EC cluster center embeddings if possible
#     if train_data == "split70":
#         emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
#     elif train_data == "split100":
#         emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
#     else:
#         # emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
#         emb_train = esm_embedding(ec_id_dict_train, device, dtype)
        
#     # emb_test = model_embedding_test(id_ec_test, model, device, dtype)
#     emb_test = model_embedding_test_from_input_embeddings(id_ec_test, model, device, dtype)
#     eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
#     seed_everything()
#     eval_df = pd.DataFrame.from_dict(eval_dist)
#     ensure_dirs("./results")
#     out_filename = "results/" +  test_data
#     write_max_sep_choices(eval_df, out_filename, gmm=gmm)
#     if report_metrics:
#         pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
#         pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
#         true_label, all_label = get_true_labels('./data/' + test_data)
#         pre, rec, f1, roc, acc = get_eval_metrics(
#             pred_label, pred_probs, true_label, all_label)
#         print("############ EC calling results using maximum separation ############")
#         print('-' * 75)
#         print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
#             f'>>> precision: {pre:.3} | recall: {rec:.3}'
#             f'| F1: {f1:.3} | AUC: {roc:.3} ')
#         print('-' * 75)


def infer_maxsep_from_embeddings(train_data, test_data, report_metrics=False, 
                 pretrained=True, model_name=None, gmm=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    
    # load checkpoints
    model = LayerNormNet(384, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = esm_embedding(ec_id_dict_train, device, dtype)
        
    emb_test = model_embedding_test_from_input_embeddings(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" + test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        
        # Create metrics dictionary
        metrics_dict = {
            'train_data': train_data,
            'test_data': test_data,
            'total_samples': len(true_label),
            'total_ec': len(all_label),
            'precision': pre,
            'recall': rec,
            'f1_score': f1,
            'auc': roc,
            'accuracy': acc,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name if model_name else 'pretrained',
            'gmm_used': gmm is not None
        }
        
        # Save metrics to CSV file
        metrics_filename = f"results/metrics_{train_data}_{test_data}.csv"
        ensure_dirs("./results")
        
        # Check if file exists, append if exists, otherwise create new file
        if os.path.exists(metrics_filename):
            existing_df = pd.read_csv(metrics_filename)
            new_df = pd.DataFrame([metrics_dict])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(metrics_filename, index=False)
        else:
            pd.DataFrame([metrics_dict]).to_csv(metrics_filename, index=False)
        
        # Save detailed prediction results
        all_predictions_df = save_detailed_predictions(pred_label, true_label, pred_probs, 
                                                      id_ec_test, train_data, test_data)
        
        # Extract correct prediction samples from all predictions
        correct_predictions_df = all_predictions_df[all_predictions_df['correct'] == True]
        correct_filename = f"results/correct_predictions_{train_data}_{test_data}.csv"
        correct_predictions_df.to_csv(correct_filename, index=False)
        
        # Also save wrong prediction samples for analysis
        wrong_predictions_df = all_predictions_df[all_predictions_df['correct'] == False]
        wrong_filename = f"results/wrong_predictions_{train_data}_{test_data}.csv"
        wrong_predictions_df.to_csv(wrong_filename, index=False)
        
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)}')
        print(f'>>> precision: {pre:.3} | recall: {rec:.3} | F1: {f1:.3} | AUC: {roc:.3} | Accuracy: {acc:.3}')
        print(f'>>> Correct predictions: {len(correct_predictions_df)}/{len(true_label)} ({len(correct_predictions_df)/len(true_label)*100:.1f}%)')
        print(f'>>> Wrong predictions: {len(wrong_predictions_df)}/{len(true_label)} ({len(wrong_predictions_df)/len(true_label)*100:.1f}%)')
        print(f'>>> Metrics saved to: {metrics_filename}')
        print(f'>>> Correct predictions saved to: {correct_filename}')
        print(f'>>> Wrong predictions saved to: {wrong_filename}')
        print(f'>>> All predictions saved to: results/all_predictions_{train_data}_{test_data}.csv')
        print('-' * 75)
        
        return metrics_dict, correct_predictions_df, wrong_predictions_df
    
    return None



def infer_get_embedding(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, output_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    # id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    id_ec_test, _ = get_ec_id_dict_non_prom_without_label('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(384, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    # if train_data == "split70":
    #     emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    # elif train_data == "split100":
    #     emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    # else:
    #     emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    # emb_train = model_embedding_test(id_ec_train, model, device, dtype)
        # Save embeddings for each ID in the train dataset
    # for idx, emb in zip(id_ec_train, emb_train):
    #     emb_file = os.path.join(output_dir, f"{idx}.pt")
    #     torch.save(emb, emb_file)
    #     print(f"Saved embedding for {idx} in {emb_file}")

    # Save embeddings for each ID in the test dataset
    for idx, emb in zip(id_ec_test, emb_test):
        emb_file = os.path.join(output_dir, f"{idx}.pt")
        torch.save(emb.cpu(), emb_file)
        print(f"Saved embedding for {idx} in {emb_file}")

    return emb_test 
    
    # eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    # seed_everything()
    # eval_df = pd.DataFrame.from_dict(eval_dist)
    # ensure_dirs("./results")
    # out_filename = "results/" +  test_data
    # write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    # if report_metrics:
    #     pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
    #     pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
    #     true_label, all_label = get_true_labels('./data/' + test_data)
    #     pre, rec, f1, roc, acc = get_eval_metrics(
    #         pred_label, pred_probs, true_label, all_label)
    #     print("############ EC calling results using maximum separation ############")
    #     print('-' * 75)
    #     print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
    #         f'>>> precision: {pre:.3} | recall: {rec:.3}'
    #         f'| F1: {f1:.3} | AUC: {roc:.3} ')
    #     print('-' * 75)

