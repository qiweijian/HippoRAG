def init_embedding_model(model_name):
    # if 'GritLM/' in model_name:
    if 'grit' in model_name.lower():
        from src.lm_wrapper.gritlm import GritWrapper
        # return GritWrapper(model_name)
        return GritWrapper("/data/qwj/model/GritLM-7B")
    elif model_name not in ['colbertv2', 'bm25']:
        from src.lm_wrapper.huggingface_util import HuggingFaceWrapper
        return HuggingFaceWrapper(model_name)  # HuggingFace model for retrieval
