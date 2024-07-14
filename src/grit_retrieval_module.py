from RetrievalModule import RetrievalModule, VECTOR_DIR
from gritlm import GritLM
from processing import processing_phrases

import os
import gc
import pickle

import faiss
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

class RetrievalModuleForGrit(RetrievalModule):
    def __init__(self, retriever_name, string_filename, query_instruction, **kwargs):
        assert retriever_name == "grit-7b"
        self.retriever_name = "grit-7b"
        self.plm = GritLM("/data/qwj/model/GritLM-7B", torch_dtype='auto', device_map="auto", mode='embedding',**kwargs)
        self.retrieval_name_dir = VECTOR_DIR + '/' + self.retriever_name.replace('/', '_').replace('.', '')+"_mean"
        if not (os.path.exists(self.retrieval_name_dir)):
            os.makedirs(self.retrieval_name_dir)
        self.query_instruction = query_instruction
        # Get previously computed vectors
        precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors(self.retrieval_name_dir)

        # Get AUI Strings to be Encoded
        string_df = pd.read_csv(string_filename, sep='\t')
        string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
        sorted_df = self.create_sorted_df(string_df.strings.values)

        missing_strings = self.find_missing_strings(sorted_df.strings.unique(), precomp_strings)

        if len(missing_strings) > 0:
            print('Encoding {} Missing Strings'.format(len(missing_strings)))
            new_vectors = self.plm.encode_corpus(missing_strings)

            precomp_strings = list(precomp_strings)
            precomp_vectors = list(precomp_vectors)

            precomp_strings.extend(list(missing_strings))
            precomp_vectors.extend(list(new_vectors))

            precomp_vectors = np.array(precomp_vectors)

            self.save_vecs(precomp_strings, precomp_vectors, self.retrieval_name_dir)

        self.vector_dict = self.make_dictionary(sorted_df, precomp_strings, precomp_vectors)

        print('Vectors Loaded.')

        queries = string_df[string_df.type == 'query']
        kb = string_df[string_df.type == 'kb']

        nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values)
        pickle.dump(nearest_neighbors, open(self.retrieval_name_dir + '/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0]), 'wb'))
    
    def encode_queries(self, query_list):
        return self.plm.encode_corpus(query_list, instruction=self.query_instruction)
    
    def retrieve_knn(self, queries, knowledge_base, k=2047):
        # all the same but the encoded query
        original_vecs = []
        new_vecs = []

        for string in knowledge_base:
            original_vecs.append(self.vector_dict[string])

        # for string in queries:
        #     new_vecs.append(self.encode_query(string))
        new_vecs = self.encode_queries(queries)

        if len(original_vecs) == 0 or len(new_vecs) == 0:
            return {}

        original_vecs = np.vstack(original_vecs)
        new_vecs = np.vstack(new_vecs)

        original_vecs = original_vecs.astype(np.float32)
        new_vecs = new_vecs.astype(np.float32)

        faiss.normalize_L2(original_vecs)
        faiss.normalize_L2(new_vecs)

        # Preparing Data for k-NN Algorithm
        print('Chunking')

        dim = len(original_vecs[0])
        index_split = 4
        index_chunks = np.array_split(original_vecs, index_split)
        query_chunks = np.array_split(new_vecs, 100)

        # Building and Querying FAISS Index by parts to keep memory usage manageable.
        print('Building Index')

        index_chunk_D = []
        index_chunk_I = []

        current_zero_index = 0

        for num, index_chunk in enumerate(index_chunks):

            print('Running Index Part {}'.format(num))
            index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)  # build the index

            if faiss.get_num_gpus() > 1:
                gpu_resources = []

                for i in range(faiss.get_num_gpus()):
                    res = faiss.StandardGpuResources()
                    gpu_resources.append(res)

                gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
            else:
                gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

            print()
            gpu_index.add(index_chunk)

            D, I = [], []

            for q in tqdm(query_chunks):
                d, i = gpu_index.search(q, k)

                i += current_zero_index

                D.append(d)
                I.append(i)

            index_chunk_D.append(D)
            index_chunk_I.append(I)

            current_zero_index += len(index_chunk)

            #             print(subprocess.check_output(['nvidia-smi']))

            del gpu_index
            del gpu_resources
            gc.collect()

        print('Combining Index Chunks')

        stacked_D = []
        stacked_I = []

        for D, I in zip(index_chunk_D, index_chunk_I):
            D = np.vstack(D)
            I = np.vstack(I)

            stacked_D.append(D)
            stacked_I.append(I)

        del index_chunk_D
        del index_chunk_I
        gc.collect()

        stacked_D = np.hstack(stacked_D)
        stacked_I = np.hstack(stacked_I)

        full_sort_I = []
        full_sort_D = []

        for d, i in tqdm(zip(stacked_D, stacked_I)):
            sort_indices = np.argsort(d, kind='stable')

            sort_indices = sort_indices[::-1]

            i = i[sort_indices][:k]
            d = d[sort_indices][:k]

            full_sort_I.append(i)
            full_sort_D.append(d)

        del stacked_D
        del stacked_I
        gc.collect()

        sorted_candidate_dictionary = {}

        for new_index, nn_info in tqdm(enumerate(zip(full_sort_I, full_sort_D))):
            nn_inds, nn_dists = nn_info
            nns = [knowledge_base[i] for i in nn_inds]

            sorted_candidate_dictionary[queries[new_index]] = (nns, nn_dists)

        return sorted_candidate_dictionary
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_name', type=str, help='retrieval model name, e.g., "facebook/contriever"')
    parser.add_argument('--string_filename', type=str, nargs='+', help='list of string filenames')
    parser.add_argument('--query_type', type=str, choices=['entity', 'both'])

    args = parser.parse_args()

    retriever_name = args.retriever_name
    # string_filename = args.string_filename
    if args.query_type == "entity":
        instruction = "Given an entity, retrieve the relevant entities"
    elif args.query_type == "both":
        instruction = "Given an entity or a relation, retrieve the relevant entities or relations"
    else:
        raise NotImplementedError
    query_instruction = "<|user|>\n" + instruction + "\n<|embed|>\n" 

    for string_filename in args.string_filename:
        retrieval_module = RetrievalModuleForGrit(retriever_name, string_filename, query_instruction)