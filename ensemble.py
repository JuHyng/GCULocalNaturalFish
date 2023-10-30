#ensemble model daekeun-ml/koelectra-small-v3-korsts
from inference_korsts import model_fn, input_fn, predict_fn, output_fn
from tqdm import tqdm

model = model_fn(None)

import json
import os, sys

def dict_to_jsonl_line(data):
    """
    Convert a dictionary to a jsonl line (a JSON string followed by a newline character).
    
    Parameters:
        - data: dict, the data to be converted.
    
    Returns:
        - str, the jsonl line.
    """
    return json.dumps(data, ensure_ascii=False) + '\n'


def get_output(jsonl_line):
    """Extracts the 'output' value from a jsonl line."""
    return json.loads(jsonl_line)['output']

def sts_similarity(sentence1, sentence2):
    """
    Compute the semantic textual similarity between two sentences.
    
    Note: You will implement this function using an STS model.
    
    Parameters:
        - sentence1, sentence2: strings to be compared.
    
    Returns:
        - float in range [0, 5], indicating the similarity between sentence1 and sentence2.
    """
    input = {'text':[sentence1, sentence2]}
    transformed_inputs = input_fn(input)
    predicted_classes = predict_fn(transformed_inputs, model)
    model_outputs = output_fn(predicted_classes)
    
    return model_outputs[0]['score']

def ensemble_voting(filepaths):
    """
    Perform ensemble voting on a list of output files. 
    
    Parameters:
        - filepaths: a list of filepaths to the output jsonl files.
    
    Returns:
        - The selected output for each example, based on STS similarity.
    """
    # Read the files into memory
    all_outputs = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_outputs.append([get_output(line) for line in f])
    
    # Ensemble voting
    selected_outputs = []
    selected_counts = [0] * len(filepaths)
    
    for idx in tqdm(range(len(all_outputs[0])), desc='Processing lines', unit='line'):
        max_similarity = -1
        best_output_idx = -1
        
        # Compare each output with all other outputs
        for i, output1 in enumerate(all_outputs):
            current_similarity = 0
            for j, output2 in enumerate(all_outputs):
                if i != j:
                    current_similarity += sts_similarity(output1[idx], output2[idx])
            
            # If the current output has greater similarity than the best so far, update
            if current_similarity > max_similarity:
                max_similarity = current_similarity
                best_output_idx = i
        
        # Add the best output to the selected outputs
        selected_outputs.append(all_outputs[best_output_idx][idx])
        # Increment the count for the selected file
        selected_counts[best_output_idx] += 1
        
    # Return dictionary for counts
    counts = dict(zip(filepaths, selected_counts))
    
    return selected_outputs, counts


def fill_outputs(test_filepath, selected_outputs, output_filepath):
    """
    Fill the original test file with the selected outputs and write to a new file.
    
    Parameters:
        - test_filepath: str, path to the original test file with empty outputs.
        - selected_outputs: list, the selected outputs to fill in.
        - output_filepath: str, path to write the new file with filled outputs.
    """
    with open(test_filepath, 'r', encoding='utf-8') as test_file, \
         open(output_filepath, 'w', encoding='utf-8') as output_file:
        
        for line, output in zip(test_file, selected_outputs):
            data = json.loads(line)
            data['output'] = output
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            
if __name__ == '__main__':
    
    test_filepath = sys.argv[1]
    load_dir = sys.argv[2]

    ensemble_files = [load_dir + '/' + file for file in os.listdir(load_dir)]
  
    output_filepath = sys.argv[3]
    
    outputs, counts = ensemble_voting(ensemble_files)

    fill_outputs(test_filepath, outputs, output_filepath)
    print(counts)

