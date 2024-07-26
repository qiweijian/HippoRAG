import sys
import os

sys.path.append('.')

import ipdb
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from typing import Any, Tuple
import json

from tqdm import tqdm

from hipporag_v3 import HippoRAGv3, HipporagConfig

# import debugpy

# # 绑定到 localhost 5679 端口
# debugpy.listen(("localhost", 5679))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


def parse_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []

    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split('\n', 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        answer = qa_pair[1].strip()

        parsed_data.append({
            'metadata': metadata,
            'document': document_text,
            'question': question,
            'answer': answer
        })

    return parsed_data


def retrieve_step(query: str, corpus, top_k: int, rag: HippoRAGv3, dataset: str):
    ranks, scores, logs = rag.rank_docs(query, top_k=top_k)
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        retrieved_passages = []
        for rank in ranks:
            key = list(corpus.keys())[rank]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    else:
        retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    return retrieved_passages, scores, logs


def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """
    prompt_demo = ''
    for sample in few_shot:
        prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print(e)
        return ''
    return response_content


@dataclass
class ircotConfig(HipporagConfig):
    prompt: str = field(
        default=None,
        metadata={'help': 'The prompt to use'}
    )
    num_demo: int = field(
        default=1,
        metadata={'help': 'The number of demo samples'}
    )
    max_steps: int = field(
        default=None,
        metadata={'help': 'Maximum number of steps'}
    )
    top_k: int = field(
        default=8,
        metadata={'help': 'Retrieving k documents at each step'}
    )
    force_retry: bool = field(
        default=False,
        metadata={'help': 'Force retry if necessary'}
    )

    @property
    def output_path(self):
        doc_ensemble_str = f'doc_ensemble_{self.recognition_threshold}' if self.doc_ensemble else 'no_ensemble'
        dpr_only_str = 'dpr_only' if self.dpr_only else 'hipporag'
        if self.graph_alg == 'ppr':
            output_path = f'output/ircot_v3/ircot_results_{self.corpus_name}_{dpr_only_str}_{self.graph_creating_retriever_name_processed}_demo_{self.num_demo}_{self.extraction_model_name_processed}_{doc_ensemble_str}_step_{self.max_steps}_top_{self.top_k}_sim_thresh_{self.sim_threshold}'
            if self.damping != 0.1:
                output_path += f'_damping_{self.damping}'
        else:
            output_path = f'output/ircot_v3/ircot_results_{self.corpus_name}_{dpr_only_str}_{self.graph_creating_retriever_name_processed}_demo_{self.num_demo}_{self.extraction_model_name_processed}_{doc_ensemble_str}_step_{self.max_steps}_top_{self.top_k}_{self.graph_alg}_sim_thresh_{self.sim_threshold}'

        if not self.node_specificity:
            output_path += 'wo_node_spec'
        output_path += '.json'
        return output_path


def get_args() -> Tuple[ircotConfig, Any]:
    parser = HfArgumentParser(ircotConfig)
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)


if __name__ == '__main__':
    ircot_config, unknown_args = get_args()
    print(ircot_config)

    rag = HippoRAGv3(ircot_config)

    data = json.load(open(ircot_config.get_path('data_path'), 'r'))
    corpus = json.load(open(ircot_config.get_path('corpus_path'), 'r'))

    if ircot_config.max_steps > 1:
        client = init_langchain_model(ircot_config.extraction_model, ircot_config.extraction_model_name)
        prompt_path = ircot_config.get_path('prompt_path')
        few_shot_samples = parse_prompt(prompt_path)[:ircot_config.num_demo]

    k_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
    total_recall = {k: 0 for k in k_list}

    os.makedirs(os.path.dirname(ircot_config.output_path), exist_ok=True)
    if ircot_config.force_retry:
        results = []
        processed_ids = set()
    else:
        try:
            with open(ircot_config.output_path, 'r') as f:
                results = json.load(f)
            if ircot_config.corpus_name in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                processed_ids = {sample['_id'] for sample in results}
            else:
                processed_ids = {sample['id'] for sample in results}

            for sample in results:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
        except Exception as e:
            print(e)
            print('Results file maybe empty, cannot be loaded.')
            results = []
            processed_ids = set()

    print(f'Loaded {len(results)} results from {ircot_config.output_path}')
    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()

    for sample_idx, sample in tqdm(enumerate(data), total=len(data), desc='IRCoT retrieval'):  # for each sample
        if ircot_config.corpus_name in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample_id = sample['_id']
        else:
            sample_id = sample['id']

        if sample_id in processed_ids:
            continue

        query = sample['question']
        all_logs = {}

        retrieved_passages, scores, logs = retrieve_step(query, corpus, ircot_config.top_k, rag, ircot_config.corpus_name)

        it = 1
        all_logs[it] = logs

        thoughts = []
        retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}

        while it < ircot_config.max_steps:  # for each iteration of IRCoT
            new_thought = reason_step(ircot_config.corpus_name, few_shot_samples, query, retrieved_passages[:ircot_config.top_k], thoughts, client)
            thoughts.append(new_thought)
            if 'So the answer is:' in new_thought:
                break
            it += 1

            new_retrieved_passages, new_scores, logs = retrieve_step(new_thought, corpus, ircot_config.top_k, rag, ircot_config.corpus_name)
            all_logs[it] = logs

            for passage, score in zip(new_retrieved_passages, new_scores):
                if passage in retrieved_passages_dict:
                    retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                else:
                    retrieved_passages_dict[passage] = score

            retrieved_passages, scores = zip(*retrieved_passages_dict.items())

            sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
            retrieved_passages, scores = zip(*sorted_passages_scores)
        # end iteration

        # calculate recall
        if ircot_config.corpus_name in ['hotpotqa', 'hotpotqa_train']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif ircot_config.corpus_name in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif ircot_config.corpus_name == "musique":
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
            retrieved_items = retrieved_passages
        else:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
            retrieved_items = retrieved_passages

        # calculate metrics
        recall = dict()
        # print(f'idx: {sample_idx + 1} ', end='')
        for k in k_list:
            recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
            total_recall[k] += recall[k]
        #     print(f'R@{k}: {total_recall[k] / (sample_idx + 1):.4f} ', end='')
        # print()
        # print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

        # record results
        phrases_in_gold_docs = []
        for gold_item in gold_items:
            phrases_in_gold_docs.append(rag.get_phrases_in_doc_str(gold_item))

        if ircot_config.corpus_name in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample['supporting_docs'] = [item for item in sample['supporting_facts']]
        else:
            sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
            del sample['paragraphs']

        sample['retrieved'] = retrieved_passages[:10]
        sample['retrieved_scores'] = scores[:10]
        sample['nodes_in_gold_doc'] = phrases_in_gold_docs
        sample['recall'] = recall
        first_log = all_logs[1]
        for key in first_log.keys():
            sample[key] = first_log[key]
        sample['thoughts'] = thoughts
        results.append(sample)

        if (sample_idx + 1) % 10 == 0:
            with open(ircot_config.output_path, 'w') as f:
                json.dump(results, f)

    # save results
    with open(ircot_config.output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved {len(results)} results to {ircot_config.output_path}')