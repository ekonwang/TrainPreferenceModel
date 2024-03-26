from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.models.modeling_bge import BGECustomEmbedder
from embedding.models.modeling_internlm import InternLMEmbedder

MODEL_MAPPING = dict(
    BERT=BertEmbedder,
    Mistral=MistralEmbedder,
    BGE=BGECustomEmbedder,
    InternLM=InternLMEmbedder
)

# __all__ = ['BertEmbedder', 'MistralEmbedder', 'BGECustomEmbedder']