import json
import logging
import os
import _pickle as pickle
from collections import defaultdict
from glob import glob

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import torch
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import RunConfig, Run, ColBERTConfig
from tqdm import tqdm
import copy
import re
from dataclasses import dataclass

from src.colbertv2_indexing import colbertv2_index
from src.langchain_util import init_langchain_model, LangChainModel
from src.lm_wrapper.util import init_embedding_model
from src.named_entity_extraction_parallel import named_entity_recognition
from src.processing import processing_phrases, min_max_normalize

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

COLBERT_CKPT_DIR = "exp/colbertv2.0"

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Union

@dataclass
class HipporagConfig:
    corpus_name: Literal['2wikihopqa', 'hotpotqa', 'musique'] = field(
        default='hotpotqa', metadata={"help": "Name of the dataset to use for retrieval"}
    )
    extraction_model: Literal['openai', 'together'] = field(
        default='openai', metadata={"help": "LLM provider for query NER, e.g., 'openai' or 'together'"}
    )
    extraction_model_name: Literal['meta-llama_Llama-3-8b-chat-hf', 'gpt-3.5-turbo-1106', 'meta-llama_Llama-3-70b-chat-hf'] = field(
        default='gpt-3.5-turbo-1106', metadata={"help": "The extraction model to be used."}
    )
    graph_creating_retriever_name: str = field(
        default='facebook/contriever', metadata={"help": "Retrieval encoder used to link query named entities with query nodes"}
    )
    extraction_type: str = field(
        default='ner', metadata={"help": "Type of NER extraction during indexing"}
    )
    graph_type: str = field(
        default='facts_and_sim', metadata={"help": "Type of graph used by HippoRAG"}
    )
    sim_threshold: float = field(
        default=0.8, metadata={"help": "Synonymy threshold which was used to create the graph that will be used by HippoRAG"}
    )
    node_specificity: bool = field(
        default=True, metadata={"help": "Flag that determines whether node specificity will be used"}
    )
    doc_ensemble: bool = field(
        default=False, metadata={"help": "Flag to determine whether to use uncertainty-based ensembling"}
    )
    colbert_config: Optional[dict] = field(
        default=None, metadata={"help": "ColBERTv2 configuration"}
    )
    dpr_only: bool = field(
        default=False, metadata={"help": "Flag to determine whether HippoRAG will be used at all"}
    )
    graph_alg: str = field(
        default='ppr', metadata={"help": "Type of graph algorithm to be used for retrieval, defaults to PPR"}
    )
    damping: float = field(
        default=0.1, metadata={"help": "Damping factor for PPR"}
    )
    recognition_threshold: float = field(
        default=0.9, metadata={"help": "Threshold used for uncertainty-based ensembling"}
    )
    corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the corpus file (see the format in README.md), not needed for now if extraction files are already present"}
    )
    linking_retriever_name: Optional[str] = field(
        default=None, metadata={"help": "Retriever name for linking, defaults to graph_creating_retriever_name if None"}
    )
    def __post_init__(self):
        if 'colbert' in self.graph_creating_retriever_name.lower() and self.colbert_config is None:
            self.colbert_config = {'root': f'data/lm_vectors/colbert/{self.corpus_name}',
                                   'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
        self.phrase_type = 'ents_only_lower_preprocess'
        self.extraction_model_name_processed = self.extraction_model_name.replace('/', '_')
        if self.linking_retriever_name is None:
            self.linking_retriever_name = self.graph_creating_retriever_name

        if self.extraction_model_name != 'gpt-3.5-turbo-1106':
            self.extraction_type = self.extraction_type + '_' + self.extraction_model_name_processed
        self.graph_creating_retriever_name_processed = self.graph_creating_retriever_name.replace('/', '_').replace('.', '')
        self.linking_retriever_name_processed = self.linking_retriever_name.replace('/', '_').replace('.', '')
        self.version = 'v3'

        self.path_dict = {
            'named_entity_cache': 'output/{}_queries.named_entity_output.tsv'.format(self.corpus_name),
            'data_path': f'data/{self.corpus_name}.json',
            'corpus_path': 'data/{}_corpus.json'.format(self.corpus_name),
            'index_file_pattern': 'output/openie_{}_results_{}_{}_*.json'.format(self.corpus_name, self.extraction_type, self.extraction_model_name_processed),
            'relation_to_id':'output/{}_{}_relation_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'kb_node_phrase_to_id':'output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'lose_fact_dict': 'output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'relations_dict': 'output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(
                    self.corpus_name, self.graph_type, self.phrase_type,
                    self.extraction_type, self.graph_creating_retriever_name_processed, self.version),
            'docs_to_facts': 'output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'facts_to_phrases': 'output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'docs_to_facts_mat': 'output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'facts_to_phrases_mat': 'output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'graph_file': 'output/{}_{}_graph_mean_{}_thresh_{}_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, str(self.sim_threshold), self.phrase_type, self.extraction_type, self.graph_creating_retriever_name_processed, self.version),
            'encoded_string': 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(self.linking_retriever_name_processed),
            'kb_node_phrase_embeddings': 'data/lm_vectors/{}_mean/{}_kb_node_phrase_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name),
            'doc_embeddings_cache': 'data/lm_vectors/{}_mean/{}_doc_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name),
            'prompt_path': self.prompt_path()
        }
    
    def prompt_path(self):
        if 'hotpotqa' in self.corpus_name:
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif 'musique' in self.corpus_name:
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif '2wikimultihopqa' in self.corpus_name:
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        else:
            prompt_path = f'data/ircot_prompts/{self.corpus_name}/gold_with_3_distractors_context_cot_qa_codex.txt'
        return prompt_path

    def get_path(self, key):
        return self.path_dict.get(key, None)
        
@dataclass
class Entity:
    name: str
    id: int
    in_relations: Dict = None
    out_relations: Dict = None

    def __post_init__(self):
        if self.in_relations is None:
            self.in_relations = defaultdict(list)
        if self.out_relations is None:
            self.out_relations = defaultdict(list)

    def __repr__(self):
        return (f"Entity(name={self.name}, id={self.id}, "
                f"{len(self.in_relations)} in_relations, "
                f"{len(self.out_relations)} out_relations")

class KnowledgeBase:
    def __init__(self, kb_phrase_dict: Dict[str, int]):
        self.entities = {id: Entity(name=name, id=id) for name, id in kb_phrase_dict.items()}
        self.name_to_id = kb_phrase_dict
    
    def get(self, identifier: Union[str, int]) -> Optional[Entity]:
        if isinstance(identifier, int):
            return self.entities.get(identifier, None)
        elif isinstance(identifier, str):
            entity_id = self.name_to_id.get(identifier, None)
            if entity_id is not None:
                return self.entities.get(entity_id, None)
        return None
    
    def add_triplet(self, head_id: int, relation_name: str, tail_id: int):
        entity1 = self.entities.get(head_id)
        entity2 = self.entities.get(tail_id)
        
        if entity1 is not None and entity2 is not None:
            entity1.out_relations[relation_name].append(entity2)
            entity2.in_relations[relation_name].append(entity1)
        else:
            raise ValueError("One or both of the entity IDs do not exist.")

        

class HippoRAGv3:
    def __init__(self, config:HipporagConfig, qa_model=None):
        self.config = config
        self.client = init_langchain_model(self.config.extraction_model, self.config.extraction_model_name)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.named_entity_cache = self.load_queries_cache()

        self.embed_model = init_embedding_model(self.config.linking_retriever_name)
        self.load_embeddings()

        self.statistics = {}
        self.ensembling_debug = []
        if qa_model is None:
            qa_model = LangChainModel('openai', 'gpt-3.5-turbo')
        self.qa_model = init_langchain_model(qa_model.provider, qa_model.model_name)

    def load_embeddings(self):
        # Loading Important Corpus Files
        if not self.config.dpr_only:
            self.load_index_files()

            # Construct Graph
            self.create_graph()

            # Loading Node Embeddings
            self.load_node_vectors()
        else:
            self.load_corpus()

        if (self.config.doc_ensemble or self.config.dpr_only) and self.config.linking_retriever_name not in ['colbertv2', 'bm25']:
            # Loading Doc Embeddings
            self.get_dpr_doc_embedding()

        if self.config.linking_retriever_name == 'colbertv2':
            if self.config.dpr_only is False or self.config.doc_ensemble:
                colbertv2_index(self.phrases.tolist(), self.config.corpus_name, 'phrase', self.config.colbert_config['phrase_index_name'], overwrite='reuse')
                with Run().context(RunConfig(nranks=1, experiment="phrase", root=self.config.colbert_config['root'])):
                    config = ColBERTConfig(root=self.config.colbert_config['root'], )
                    self.phrase_searcher = Searcher(index=self.config.colbert_config['phrase_index_name'], config=config, verbose=0)
            if self.config.doc_ensemble or self.config.dpr_only:
                colbertv2_index(self.dataset_df['paragraph'].tolist(), self.config.corpus_name, 'corpus', self.config.colbert_config['doc_index_name'], overwrite='reuse')
                with Run().context(RunConfig(nranks=1, experiment="corpus", root=self.config.colbert_config['root'])):
                    config = ColBERTConfig(root=self.config.colbert_config['root'], )
                    self.corpus_searcher = Searcher(index=self.config.colbert_config['doc_index_name'], config=config, verbose=0)


    def load_queries_cache(self) -> Dict:
        try:
            named_entity_cache = pd.read_csv(self.config.get_path('named_entity_cache'), sep='\t')
        except Exception as e:
            named_entity_cache = pd.DataFrame([], columns=['query', 'triples'])

        if 'query' in named_entity_cache:
            named_entity_cache = {row['query']: eval(row['triples']) for i, row in
                                       named_entity_cache.iterrows()}
        elif 'question' in named_entity_cache:
            named_entity_cache = {row['question']: eval(row['triples']) for i, row in named_entity_cache.iterrows()}
        return named_entity_cache
        

    def get_passage_by_idx(self, passage_idx):
        """
        Get the passage by its index
        @param passage_idx: the index of the passage
        @return: the passage
        """
        return self.dataset_df.iloc[passage_idx]['paragraph']

    def get_extraction_by_passage_idx(self, passage_idx, chunk=False):
        """
        Get the extraction results for a specific passage.
        @param passage_idx: the passage idx, i.e., 'idx' within each passage dict, not the array index for the corpus
        @param chunk: whether the corpus is chunked
        @return: the extraction results for the passage
        """
        # find item with idx == passage_idx in self.extracted_triples
        for item in self.extracted_triples:
            if not chunk and item['idx'] == passage_idx:
                return item
            elif chunk and (item['idx'] == passage_idx or item['idx'].startswith(passage_idx + '_')):
                return item
        return None

    def get_shortest_distance_between_nodes(self, node1: str, node2: str):
        """
        Get the shortest distance between two nodes in the graph
        @param node1: node1 phrase
        @param node2: node2 phrase
        @return: the shortest distance between the two nodes
        """
        try:
            node1_id = np.where(self.phrases == node1)[0][0]
            node2_id = np.where(self.phrases == node2)[0][0]

            return self.g.shortest_paths(node1_id, node2_id)[0][0]
        except Exception as e:
            return -1

    def rank_docs(self, query: str, top_k=10):
        """
        Rank documents based on the query
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores
        """

        assert isinstance(query, str), 'Query must be a string'
        query_ner_list = self.query_ner(query)

        if 'colbertv2' in self.config.linking_retriever_name:
            # Get Query Doc Scores
            queries = Queries(path=None, data={0: query})
            if self.config.doc_ensemble:
                query_doc_scores = np.zeros(self.doc_to_phrases_mat.shape[0])
                ranking = self.corpus_searcher.search_all(queries, k=self.doc_to_phrases_mat.shape[0])
                # max_query_score = self.get_colbert_max_score(query)
                for doc_id, rank, score in ranking.data[0]:
                    query_doc_scores[doc_id] = score
            elif self.config.dpr_only:
                query_doc_scores = np.zeros(len(self.dataset_df))
                ranking = self.corpus_searcher.search_all(queries, k=len(self.dataset_df))
                for doc_id, rank, score in ranking.data[0]:
                    query_doc_scores[doc_id] = score

            if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
                all_phrase_weights, linking_score_map = self.link_node_by_colbertv2(query_ner_list)
        else:  # dense retrieval model
            # Get Query Doc Scores
            if self.config.doc_ensemble or self.config.dpr_only:
                query_embedding = self.embed_model.encode_text(query, return_cpu=True, return_numpy=True, norm=True)
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]

            if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
                all_phrase_weights, linking_score_map = self.link_node_by_dpr(query_ner_list)

        # Run Personalized PageRank (PPR) or other Graph Algorithm Doc Scores
        if not self.config.dpr_only:
            if len(query_ner_list) > 0:
                combined_vector = np.max([all_phrase_weights], axis=0)

                if self.config.graph_alg == 'ppr':
                    ppr_phrase_probs = self.run_pagerank_igraph_chunk([all_phrase_weights])[0]
                elif self.config.graph_alg == 'none':
                    ppr_phrase_probs = combined_vector
                elif self.config.graph_alg == 'neighbor_2':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 2)
                elif self.config.graph_alg == 'neighbor_3':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                elif self.config.graph_alg == 'paths':
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                else:
                    assert False, f'Graph Algorithm {self.config.graph_alg} Not Implemented'

                fact_prob = self.facts_to_phrases_mat.dot(ppr_phrase_probs)
                ppr_doc_prob = self.docs_to_facts_mat.dot(fact_prob)
                ppr_doc_prob = min_max_normalize(ppr_doc_prob)
            else:  # dpr_only or no entities found
                ppr_doc_prob = np.ones(len(self.extracted_triples)) / len(self.extracted_triples)

        # Combine Query-Doc and PPR Scores
        if self.config.doc_ensemble or self.config.dpr_only:
            # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
            if len(query_ner_list) == 0:
                doc_prob = query_doc_scores
                self.statistics['doc'] = self.statistics.get('doc', 0) + 1
            elif np.min(list(linking_score_map.values())) > self.config.recognition_threshold:  # high confidence in named entities
                doc_prob = ppr_doc_prob
                self.statistics['ppr'] = self.statistics.get('ppr', 0) + 1
            else:  # relatively low confidence in named entities, combine the two scores
                # the higher threshold, the higher chance to use the doc ensemble
                doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
                query_doc_scores = min_max_normalize(query_doc_scores)

                top_ppr = np.argsort(ppr_doc_prob)[::-1][:10]
                top_ppr = [(top, ppr_doc_prob[top]) for top in top_ppr]

                top_doc = np.argsort(query_doc_scores)[::-1][:10]
                top_doc = [(top, query_doc_scores[top]) for top in top_doc]

                top_hybrid = np.argsort(doc_prob)[::-1][:10]
                top_hybrid = [(top, doc_prob[top]) for top in top_hybrid]

                self.ensembling_debug.append((top_ppr, top_doc, top_hybrid))
                self.statistics['ppr_doc_ensemble'] = self.statistics.get('ppr_doc_ensemble', 0) + 1
        else:
            doc_prob = ppr_doc_prob

        # Return ranked docs and ranked scores
        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]

        if not (self.config.dpr_only) and len(query_ner_list) > 0:
            # logs
            phrase_one_hop_triples = []
            for phrase_id in np.where(all_phrase_weights > 0)[0]:
                # get all the triples that contain the phrase from self.graph_plus
                for t in list(self.kg_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1]])
                for t in list(self.kg_inverse_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1], 'inv'])

            # get top ranked nodes from doc_prob and self.doc_to_phrases_mat
            nodes_in_retrieved_doc = []
            for doc_id in sorted_doc_ids[:5]:
                node_id_in_doc = list(np.where(self.doc_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
                nodes_in_retrieved_doc.append([self.phrases[node_id] for node_id in node_id_in_doc])

            # get top ppr_phrase_probs
            top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]

            # get phrases for top_pagerank_phrase_ids
            top_ranked_nodes = [self.phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]
            logs = {'named_entities': query_ner_list, 'linked_node_scores': [list(k) + [float(v)] for k, v in linking_score_map.items()],
                    '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                    'top_ranked_nodes': top_ranked_nodes, 'nodes_in_retrieved_doc': nodes_in_retrieved_doc}
        else:
            logs = {}

        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs

    def query_ner(self, query):
        if self.config.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities
            try:
                if query in self.named_entity_cache:
                    query_ner_list = self.named_entity_cache[query]['named_entities']
                else:
                    query_ner_json, total_tokens = named_entity_recognition(self.client, query)
                    query_ner_list = eval(query_ner_json)['named_entities']

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except:
                self.logger.error('Error in Query NER')
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector, max_depth=1):

        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def load_corpus(self):
        if self.corpus_path is None:
            self.corpus_path = self.config.get_path('corpus_path')
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        self.corpus = json.load(open(self.corpus_path, 'r'))
        self.dataset_df = pd.DataFrame()
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]

    def load_openie(self):
        index_file_pattern = self.config.get_path('index_file_pattern')
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(f'No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done.')
            return
        max_samples = np.max(
            [int(file.split('{}_'.format(self.config.extraction_model_name_processed))[1].split('.json')[0]) for file in possible_files])
        extracted_file = json.load(open(
            index_file_pattern[::-1].replace('*', str(max_samples)[::-1], 1)[::-1],
            'r'))

        self.extracted_triples = extracted_file['docs']

        if self.config.corpus_name == 'hotpotqa':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        if self.config.corpus_name == 'hotpotqa_train':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif 'musique' in self.config.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif self.config.corpus_name == '2wikimultihopqa':
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
            self.dataset_df['title'] = [s['title'] for s in self.extracted_triples]
        elif 'case_study' in self.config.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        else:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]

    def load_index_files(self):
        self.load_openie()
        
        self.kb_node_phrase_to_id = pickle.load(open(self.config.get_path('kb_node_phrase_to_id'), 'rb'))
        self.relation_to_id = pickle.load(open(self.config.get_path('relation_to_id'), 'rb'))

        self.lose_fact_dict = pickle.load(open(self.config.get_path('lose_fact_dict'), 'rb'))

        try:
            self.relations_dict = pickle.load(open(self.config.get_path('relations_dict'), 'rb'))
        except:
            pass

        self.lose_facts = list(self.lose_fact_dict.keys())
        self.lose_facts = [self.lose_facts[i] for i in np.argsort(list(self.lose_fact_dict.values()))]
        self.phrases = np.array(list(self.kb_node_phrase_to_id.keys()))[np.argsort(list(self.kb_node_phrase_to_id.values()))]
        self.relations = np.array(list(self.relation_to_id.keys()))[np.argsort(list(self.relation_to_id.values()))]


        self.docs_to_facts = pickle.load(open(self.config.get_path('docs_to_facts'), 'rb'))
        self.facts_to_phrases = pickle.load(open(self.config.get_path('facts_to_phrases'), 'rb'))
        self.docs_to_facts_mat = pickle.load(open(self.config.get_path('docs_to_facts_mat'), 'rb'))
        self.facts_to_phrases_mat = pickle.load(open(self.config.get_path('facts_to_phrases_mat'), 'rb'))
        self.lose_fact_dict = pickle.load(open(self.config.get_path('lose_fact_dict'), 'rb'))
        self.doc_to_phrases_mat = self.docs_to_facts_mat.dot(self.facts_to_phrases_mat)
        self.doc_to_phrases_mat[self.doc_to_phrases_mat.nonzero()] = 1
        self.phrase_to_num_doc = self.doc_to_phrases_mat.sum(0).T


        self.doc_to_relation_mat = self.doc_to_relation_mat.dot(self.facts_to_phrases_mat)
        self.doc_to_relation_mat[self.doc_to_relation_mat.nonzero()] = 1
        self.relation_to_num_doc = self.doc_to_relation_mat.sum(0).T

        # graph_file_path = self.config.get_path('graph_file')
        # if os.path.isfile(graph_file_path):
        #     self.graph_plus = pickle.load(open(graph_file_path, 'rb'))  # (phrase1 id, phrase2 id) -> the number of occurrences
        # else:
        #     self.logger.exception('Graph file not found: ' + graph_file_path)

    def get_phrases_in_doc_str(self, doc: str):
        # find doc id from self.dataset_df
        try:
            doc_id = self.dataset_df[self.dataset_df.paragraph == doc].index[0]
            phrase_ids = self.doc_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            return [self.phrases[phrase_id] for phrase_id in phrase_ids]
        except:
            return []
        
    def create_graph(self):
        inter_triple_weight = 1.0
        similarity_max = 1.0
        # read lose fact dict from file
        # lose_fact_dict = pickle.load(open(self.config.get_path('lose_fact_dict'), 'rb'))
        # docs_to_facts = pickle.load(open(self.config.get_path('docs_to_facts'), 'rb'))
        self.kb_phrase_dict = pickle.load(open(self.config.get_path('kb_node_phrase_to_id'), 'rb'))
        # facts_to_phrases = pickle.load(open(self.config.get_path('facts_to_phrases'), 'rb'))

        graph, graph_json = {}, {}
        self.kb = KnowledgeBase(self.kb_phrase_dict)

        for triple, fact_id in self.lose_fact_dict.items():

            if len(triple) == 3:
                relation = triple[1]
                triple = np.array(triple)[[0, 2]]

                # docs_to_facts[(doc_id, fact_id)] = 1

                for i, phrase in enumerate(triple):
                    phrase_id = self.kb_phrase_dict[phrase]
                    # doc_phrases.append(phrase_id)

                    # facts_to_phrases[(fact_id, phrase_id)] = 1

                    for phrase2 in triple[i + 1:]:
                        
                        phrase2_id = self.kb_phrase_dict[phrase2]

                        self.kb.add_triplet(phrase_id, relation, phrase2_id)

                        fact_edge_r = (phrase_id, phrase2_id)
                        fact_edge_l = (phrase2_id, phrase_id)

                        # fact_edges.append(fact_edge_r)
                        # fact_edges.append(fact_edge_l)

                        graph[fact_edge_r] = graph.get(fact_edge_r, 0.0) + inter_triple_weight
                        graph[fact_edge_l] = graph.get(fact_edge_l, 0.0) + inter_triple_weight

                        phrase_edges = graph_json.get(phrase, {})
                        edge = phrase_edges.get(phrase2, ('triple', 0))
                        phrase_edges[phrase2] = ('triple', edge[1] + 1)
                        graph_json[phrase] = phrase_edges

                        phrase_edges = graph_json.get(phrase2, {})
                        edge = phrase_edges.get(phrase, ('triple', 0))
                        phrase_edges[phrase] = ('triple', edge[1] + 1)
                        graph_json[phrase2] = phrase_edges

                        # num_triple_edges += 1

        if 'colbert' in self.config.graph_creating_retriever_name_processed:
            kb_similarity = pickle.load(open('data/lm_vectors/colbert/nearest_neighbor_kb_to_kb.p'.format(self.config.graph_creating_retriever_name_processed), 'rb'))
        else:
            kb_similarity = pickle.load(open('data/lm_vectors/{}_mean/nearest_neighbor_kb_to_kb.p'.format(self.config.graph_creating_retriever_name_processed), 'rb'))

        print('Augmenting Graph from Similarity')

        graph_plus = copy.deepcopy(graph)

        kb_similarity = {processing_phrases(k): v for k, v in kb_similarity.items()}
        self.similar_entities_dict = defaultdict(set)
        # synonym_candidates = []

        for phrase in tqdm(kb_similarity.keys(), total=len(kb_similarity)):

            synonyms = []

            if len(re.sub('[^A-Za-z0-9]', '', phrase)) > 2:
                phrase_id = self.kb_phrase_dict.get(phrase, None)

                if phrase_id is not None:

                    nns = kb_similarity[phrase]

                    num_nns = 0
                    for nn, score in zip(nns[0], nns[1]):
                        nn = processing_phrases(nn)
                        if score < self.config.sim_threshold or num_nns > 100:
                            break

                        if nn != phrase:

                            phrase2_id = self.kb_phrase_dict.get(nn)

                            if phrase2_id is not None:
                                phrase2 = nn

                                sim_edge = (phrase_id, phrase2_id)
                                synonyms.append((nn, score))

                                self.similar_entities_dict[phrase_id].add(phrase2_id)
                                graph_plus[sim_edge] = similarity_max * score

                                num_nns += 1

                                phrase_edges = graph_json.get(phrase, {})
                                edge = phrase_edges.get(phrase2, ('similarity', 0))
                                if edge[0] == 'similarity':
                                    phrase_edges[phrase2] = ('similarity', edge[1] + score)
                                    graph_json[phrase] = phrase_edges

        edges = set()
        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)
        for edge, weight in tqdm(graph_plus.items(), total=len(graph_plus), desc='Building Graph'):
            edge1 = edge[0]
            edge2 = edge[1]

            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = graph_plus[(edge[0], edge[1])]

        # self.graph_plus = new_graph_plus
        self.g = nx.Graph()
        n_vertices = len(self.kb_node_phrase_to_id)
        self.g.add_nodes_from(range(n_vertices))

        edges = list(edges)
        self.g.add_edges_from(edges)
        weights = {(v1, v3): new_graph_plus[(v1, v3)] for v1, v3 in edges}
        nx.set_edge_attributes(self.g, weights, 'weight')
        self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')
        

    # def build_graph(self):

    #     edges = set()

    #     new_graph_plus = {}
    #     self.kg_adj_list = defaultdict(dict)
    #     self.kg_inverse_adj_list = defaultdict(dict)

    #     for edge, weight in tqdm(self.graph_plus.items(), total=len(self.graph_plus), desc='Building Graph'):
    #         edge1 = edge[0]
    #         edge2 = edge[1]

    #         if (edge1, edge2) not in edges and edge1 != edge2:
    #             new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
    #             edges.add((edge1, edge2))
    #             self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
    #             self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[(edge[0], edge[1])]

    #     self.graph_plus = new_graph_plus

    #     edges = list(edges)

    #     n_vertices = len(self.kb_node_phrase_to_id)
    #     # self.g = ig.Graph(n_vertices, edges)

    #     # self.g.es['weight'] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
    #     self.g = nx.Graph()
    #     self.g.add_nodes_from(range(n_vertices))
    #     self.g.add_edges_from(edges)
    #     weights = {(v1, v3): self.graph_plus[(v1, v3)] for v1, v3 in edges}
    #     nx.set_edge_attributes(self.g, weights, 'weight')
    #     self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')

    def load_node_vectors(self):
        encoded_string_path = self.config.get_path('encoded_string')
        if os.path.isfile(encoded_string_path):
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)
        else:  # use another way to load node vectors
            if self.config.linking_retriever_name == 'colbertv2':
                return
            kb_node_phrase_embeddings_path = self.config.get_path('kb_node_phrase_embeddings')
            if os.path.isfile(kb_node_phrase_embeddings_path):
                self.kb_node_phrase_embeddings = pickle.load(open(kb_node_phrase_embeddings_path, 'rb'))
                if len(self.kb_node_phrase_embeddings.shape) == 3:
                    self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
                self.logger.info('Loaded phrase embeddings from: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))
            else:
                self.kb_node_phrase_embeddings = self.embed_model.encode_text(self.phrases.tolist(), return_cpu=True, return_numpy=True, norm=True)
                pickle.dump(self.kb_node_phrase_embeddings, open(kb_node_phrase_embeddings_path, 'wb'))
                self.logger.info('Saved phrase embeddings to: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))

    def load_node_vectors_from_string_encoding_cache(self, string_file_path):
        self.logger.info('Loading node vectors from: ' + string_file_path)
        kb_vectors = []
        self.strings = open(string_file_path, 'r').readlines()
        vecs_path_pattern = 'data/lm_vectors/{}_mean/vecs_*'.format(self.config.linking_retriever_name_processed)
        vecs_paths = glob(vecs_path_pattern)
        for i in range(len(vecs_paths)):
            vec_path = 'data/lm_vectors/{}_mean/vecs_{}.p'.format(self.config.linking_retriever_name_processed, i)
            kb_vectors.append(torch.Tensor(pickle.load(open(vec_path, 'rb'))))
        kb_mat = torch.cat(kb_vectors)  # a matrix of phrase vectors
        self.strings = [s.strip() for s in self.strings]
        self.string_to_id = {string: i for i, string in enumerate(self.strings)}
        kb_mat = kb_mat.T.divide(torch.linalg.norm(kb_mat, dim=1)).T
        kb_mat = kb_mat.to('cuda')
        kb_only_indices = []
        num_non_vector_phrases = 0
        for i in range(len(self.kb_node_phrase_to_id)):
            phrase = self.phrases[i]
            if phrase not in self.string_to_id:
                num_non_vector_phrases += 1

            phrase_id = self.string_to_id.get(phrase, 0)
            kb_only_indices.append(phrase_id)
        self.kb_node_phrase_embeddings = kb_mat[kb_only_indices]  # a matrix of phrase vectors
        self.kb_node_phrase_embeddings = self.kb_node_phrase_embeddings.cpu().numpy()
        self.logger.info('{} phrases did not have vectors.'.format(num_non_vector_phrases))

    def get_dpr_doc_embedding(self):
        cache_filename = self.config.get_path('doc_embeddings_cache')
        if os.path.exists(cache_filename):
            self.doc_embedding_mat = pickle.load(open(cache_filename, 'rb'))
            self.logger.info(f'Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}')
        else:
            self.doc_embeddings = []
            self.doc_embedding_mat = self.embed_model.encode_text(self.dataset_df['paragraph'].tolist(), return_cpu=True, return_numpy=True, norm=True)
            pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
            self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')

    def run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run pagerank on the graph
        :param reset_prob_chunk:
        :return: PageRank probabilities
        """
        pageranked_probabilities = []

        for reset_prob in reset_prob_chunk:
            # pageranked_probs = self.g.personalized_pagerank(vertices=range(len(self.kb_node_phrase_to_id)), damping=self.config.damping, directed=False,
            #                                                 weights='weight', reset=reset_prob, implementation='prpack')
            reset_prob = {i: prob for i, prob in enumerate(reset_prob)}
            pageranked_probs = nx.pagerank(
                self.g, alpha=self.config.damping, personalization=reset_prob,
            )
            pageranked_probabilities.append(np.array(list(pageranked_probs.values())))

        return np.array(pageranked_probabilities)

    def get_colbert_max_score(self, query, searcher=None):
        if searcher is None:
            searcher = self.phrase_searcher
        queries_ = [query]
        encoded_query = searcher.encode(queries_, full_length_search=False)
        encoded_doc = searcher.checkpoint.docFromText(queries_).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()
        return max_score

    def get_colbert_real_score(self, query, doc, searcher=None):
        if searcher is None:
            searcher = self.phrase_searcher
        queries_ = [query]
        encoded_query = searcher.encode(queries_, full_length_search=False)

        docs_ = [doc]
        encoded_doc = searcher.checkpoint.docFromText(docs_).float()

        real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return real_score

    def link_node_by_colbertv2(self, query_ner_list):
        phrase_ids = []
        max_scores = []

        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

            max_score = self.get_colbert_max_score(query)

            ranking = self.phrase_searcher.search_all(queries, k=1)
            for phrase_id, rank, score in ranking.data[0]:
                phrase = self.phrases[phrase_id]
                phrases_ = [phrase]
                encoded_doc = self.phrase_searcher.checkpoint.docFromText(phrases_).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

                phrase_ids.append(phrase_id)
                max_scores.append(real_score / max_score)

        # create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        top_phrase_vec = np.zeros(len(self.phrases))

        for phrase_id in phrase_ids:
            if self.config.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                top_phrase_vec[phrase_id] = 1.0

        return top_phrase_vec, {(query, self.phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, query_ner_list)}

    def link_node_by_dpr(self, query_ner_list: list):
        """
        Get the most similar phrases (as vector) in the KG given the named entities
        :param query_ner_list:
        :return:
        """
        query_ner_embeddings = self.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)

        # Get Closest Entity Nodes
        prob_vectors = np.dot(query_ner_embeddings, self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_phrase_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity
            linked_phrase_ids.append(phrase_id)
            max_scores.append(prob_vector[phrase_id])

        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_phrase_weights = np.zeros(len(self.phrases))

        for phrase_id in linked_phrase_ids:
            if self.config.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.phrase_to_num_doc[phrase_id]

                all_phrase_weights[phrase_id] = weight
            else:
                all_phrase_weights[phrase_id] = 1.0

        linking_score_map = {(query_phrase, self.phrases[linked_phrase_id]): max_score
                             for linked_phrase_id, max_score, query_phrase in zip(linked_phrase_ids, max_scores, query_ner_list)}
        return all_phrase_weights, linking_score_map
    
    def link_relation_by_colbertv2(self, query_ner_list):
        relation_ids = []
        max_scores = []

        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            encoded_query = self.relation_searcher.encode(queries_, full_length_search=False)

            max_score = self.get_colbert_max_score(query, searcher=self.relation_searcher)

            ranking = self.relation_searcher.search_all(queries, k=1)
            for relation_id, rank, score in ranking.data[0]:
                relation = self.relations[relation_id]
                relations_ = [relation]
                encoded_doc = self.relation_searcher.checkpoint.docFromText(relations_).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

                relation_ids.append(relation_id)
                max_scores.append(real_score / max_score)

        # create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        top_relation_vec = np.zeros(len(self.relations))

        for relation_id in relation_ids:
            if self.config.node_specificity:
                if self.relation_to_num_doc[relation_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.relation_to_num_doc[relation_id]
                top_relation_vec[relation_id] = weight
            else:
                top_relation_vec[relation_id] = 1.0

        return top_relation_vec, {(query, self.relations[relation_id]): max_score for relation_id, max_score, query in zip(relation_ids, max_scores, query_ner_list)}

    def link_relation_by_dpr(self, query_ner_list: list):
        """
        Get the most similar phrases (as vector) in the KG given the named entities
        :param query_ner_list:
        :return:
        """
        query_ner_embeddings = self.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)

        # Get Closest Entity Nodes
        prob_vectors = np.dot(query_ner_embeddings, self.kb_relation_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_relation_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            relation_id = np.argmax(prob_vector)  # the phrase with the highest similarity
            linked_relation_ids.append(relation_id)
            max_scores.append(prob_vector[relation_id])

        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_relation_weights = np.zeros(len(self.relations))

        for relation_id in linked_relation_ids:
            if self.config.node_specificity:
                if self.relation_to_num_doc[relation_id] == 0:  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.relation_to_num_doc[relation_id]

                all_relation_weights[relation_id] = weight
            else:
                all_relation_weights[relation_id] = 1.0

        linking_score_map = {(query_relation, self.phrases[linked_relation_id]): max_score
                             for linked_relation_id, max_score, query_relation in zip(linked_relation_ids, max_scores, query_ner_list)}
        return all_relation_weights, linking_score_map