from typing import List, Tuple, Dict, Union, Literal, Any, TypedDict
import pickle
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass
class ExperimentConfig:
    dataset: Literal['2wikihopqa', 'hotpotqa', 'musique'] = field(
        default='musique', metadata={"help": "The dataset to be used for the experiment."}
    )
    graph_type: Literal["facts_and_sim", "facts"] = "facts_and_sim"
    retrieval_model_name: Literal['colbertv2', 'facebook/contriever'] = field(
        default='colbertv2', metadata={"help": "The retrieval model to be used."}
    )
    extraction_model_name: Literal['meta-llama_Llama-3-8b-chat-hf', 'gpt-3.5-turbo-1106', 'meta-llama_Llama-3-70b-chat-hf'] = field(
        default='gpt-3.5-turbo-1106', metadata={"help": "The extraction model to be used."}
    )
    base_dir: str = "./output"

    def __post_init__(self):
        self.extraction_model_name_processed = self.extraction_model_name.replace('/', '_')
        self.retrieval_model_name_processed = self.retrieval_model_name.replace('/', '_').replace('.', '')

        self.phrase_type: str ="ents_only_lower_preprocess"
        self.extraction_type: str = 'ner'
        self.version = "v3"

        self.path_dict = {
            'triplet_to_id': f"{self.base_dir}/{self.dataset}_{self.graph_type}_graph_fact_dict_{self.phrase_type}_{self.extraction_type}.{self.version}.subset.p",
            'entity_to_id': f"{self.base_dir}/{self.dataset}_{self.graph_type}_graph_phrase_dict_{self.phrase_type}_{self.extraction_type}.{self.version}.subset.p",
            'relation_dict': f"{self.base_dir}/{self.dataset}_{self.graph_type}_graph_relation_dict_{self.phrase_type}_{self.extraction_type}_{self.retrieval_model_name_processed}.{self.version}.subset.p"
        }

    def get_path(self, key: Literal['triplet_to_id', 'entity_to_id', 'relation_dict']) -> str:
        # Return the path based on the provided key
        return self.path_dict.get(key, "Key not found")


class TargetEntity:
    def __init__(self, entity_id: str, entity: str, direction: Literal['forward', 'backward'], relation_id: str, relation: str):
        self.entity_id = entity_id
        self.entity = entity
        self.direction = direction
        self.relation_id = relation_id
        self.relation = relation

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TargetEntity):
            return self.entity_id == other.entity_id and self.relation_id == other.relation_id and self.direction == other.direction
        return False

    def __hash__(self) -> int:
        return hash((self.direction, self.entity_id, self.relation_id))

    def to_dict(self) -> Dict[str, str]:
        return {
            'entity_id': self.entity_id,
            'entity': self.entity,
            'direction': self.direction,
            'relation_id': self.relation_id,
            'relation': self.relation
        }

    def __repr__(self) -> str:
        if self.direction == 'forward':
            return f"TargetEntity(-- {self.relation} -> {self.entity})"
        else:
            return f"TargetEntity(<- {self.relation} -- {self.entity})"

class Entity:
    def __init__(self, name: str):
        self.name = name
        self.relations: set[TargetEntity] = set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'relations': [rel.to_dict() for rel in self.relations]
        }
    
    def __repr__(self) -> str:
        return f"Entity(name={self.name}, {len(self.relations)} relations)"

class KnowledgeBase:
    def __init__(self, 
                 entities_to_id: Dict[str, int],
                 triplets_to_id: Dict[Tuple[str, str, str], int],
    ):
        self.entities_to_id = entities_to_id
        self.id_to_entities = {i: e for e, i in self.entities_to_id.items()}

        self.triplets_to_id = triplets_to_id
        self.id_to_triplets = {i: t for t, i in self.triplets_to_id.items()}

        # build relation
        all_relations = set([t[1] for t in self.triplets_to_id.keys()])
        self.relations_to_id = {r: i for i, r in enumerate(all_relations)}
        self.id_to_relations = {i: r for r, i in self.relations_to_id.items()}

        self.build_kb()

    @classmethod
    def build_from_config(cls, config: ExperimentConfig):
        read_pickle = lambda path: pickle.load(open(path, 'rb'))
        triplet_to_id: Dict[Tuple[str, str, str], int] = read_pickle(config.get_path('triplet_to_id'))
        entity_to_id: Dict[str, int] = read_pickle(config.get_path('entity_to_id'))
        relation_dict: Dict[Tuple[str, str], str] = read_pickle(config.get_path('relation_dict'))

        for (head, tail), rel in relation_dict.items():
            triplet = (head, rel, tail)
            if triplet not in triplet_to_id:
                triplet_to_id[triplet] = len(triplet_to_id)
        
        return cls(entity_to_id, triplet_to_id)
    
    def build_kb(self):
        self.kb: Dict[str, Entity] = {}
        for (head, relation, tail), _ in tqdm(self.triplets_to_id.items(), total=self.num_triplets, desc='building knowledge graph'):
            head_id = self.entities_to_id[head]
            tail_id = self.entities_to_id[tail]
            relation_id = self.relations_to_id[relation]

            head_entity_in_kb = self.kb.get(head_id, Entity(name=head))
            tail_entity_in_kb = self.kb.get(tail_id, Entity(name=tail))

            head_entity_in_kb.relations.add(TargetEntity(
                entity_id=tail_id,
                entity=tail,
                direction='forward',
                relation_id=relation_id,
                relation=relation
            ))

            tail_entity_in_kb.relations.add(TargetEntity(
                entity_id=head_id,
                entity=head,
                direction='backward',
                relation_id=relation_id,
                relation=relation
            ))
            self.kb[head_id] = head_entity_in_kb
            self.kb[tail_id] = tail_entity_in_kb

    @property
    def num_triplets(self):
        return len(self.triplets_to_id)
    
    @property
    def num_entities(self):
        return len(self.entities_to_id)
    
    @property
    def num_relations(self):
        return len(self.relations_to_id)

    def __repr__(self) -> str:
        return f"KnowledgeBase({self.num_entities} entities, {self.num_relations} relations, {self.num_triplets} triplets)"
