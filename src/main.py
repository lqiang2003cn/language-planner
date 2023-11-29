import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json

device = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device)

source = 'openai'  # select from ['openai', 'huggingface']
planning_lm_id = 'gpt2-large'  # see comments above for all options
translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
sampling_params = {
    "max_tokens": 10,
    "temperature": 0.6,
    "top_p": 0.9,
    "n": 10,
    "logprobs": 1,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.3,
    "stop": '\n'
}


def lm_engine(source, planning_lm_id, device):
    def _generate(prompt, sampling_params):
        response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
        generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
        # calculate mean log prob across tokens
        mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate


generator = lm_engine(source, planning_lm_id, device)

# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

# create action embeddings using Translated LM
with open('available_actions.json', 'r') as f:
    action_list = json.load(f)
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)

# create example task embeddings using Translated LM
with open('available_examples.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name


# example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True,
#                                                device=device)  # lower batch_size if limited by GPU memory


# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score


generator = lm_engine(source, planning_lm_id, device)

# define query task
task = 'Make breakfast'
# find most relevant example
example_idx, _ = find_most_similar(task, example_task_embedding)
example = available_examples[example_idx]
# construct initial prompt
curr_prompt = f'{example}\n\nTask: {task}'
# print example and query task
print('-' * 10 + ' GIVEN EXAMPLE ' + '-' * 10)
print(example)
print('-' * 10 + ' EXAMPLE END ' + '-' * 10)
print(f'\nTask: {task}')
for step in range(1, MAX_STEPS + 1):
    best_overall_score = -np.inf
    # query Planning LM for single-step action candidates
    samples, log_probs = generator(curr_prompt + f'\nStep {step}:', sampling_params)
    for sample, log_prob in zip(samples, log_probs):
        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
        # rank each sample by its similarity score and likelihood score
        overall_score = matching_score + BETA * log_prob
        translated_action = action_list[most_similar_idx]
        # heuristic for penalizing generating the same action as the last action
        if step > 1 and translated_action == previous_action:
            overall_score -= 0.5
        # find the translated action with highest overall score
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_action = translated_action

    # terminate early when either the following is true:
    # 1. top P*100% of samples are all 0-length (ranked by log prob)
    # 2. overall score is below CUTOFF_THRESHOLD
    # else: autoregressive generation based on previously translated action
    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
    below_threshold = best_overall_score < CUTOFF_THRESHOLD
    if are_zero_length:
        print(f'\n[Terminating early because top {P * 100}% of samples are all 0-length]')
        break
    elif below_threshold:
        print(
            f'\n[Terminating early because best overall score is lower than CUTOFF_THRESHOLD ({best_overall_score} < {CUTOFF_THRESHOLD})]')
        break
    else:
        previous_action = best_action
        formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ')  # 'open_fridge' -> 'Open fridge'
        curr_prompt += f'\nStep {step}: {formatted_action}'
        print(f'Step {step}: {formatted_action}')
