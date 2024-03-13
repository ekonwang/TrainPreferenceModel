import argparse
from train import train_embedder
from eval import evaluate_embedder
from predict import predict_embedder

parser = argparse.ArgumentParser(description='Embedder Training & Evaluation Configuration')
subparsers = parser.add_subparsers(help='Embedder Training & Evaluation')

training_parser = subparsers.add_parser(name='train', help='training embedder')
training_parser.add_argument('--embedder_name', type=str, default='mistral_embedder', help='The name of the training embedder')
training_parser.add_argument('--backbone_type', type=str, default='Mistral', help='Supported backbone types: [Mistral, BERT]')
training_parser.add_argument('--init_backbone', type=str, default='/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e', help='The parameter path of initial embedder backbone')
training_parser.add_argument('--pool_type', type=str, default='position_weight', help='Supported pool types: [position_weight, mean, eos, cls]')
training_parser.add_argument('--peft_lora', action='store_true', default=False, help='Training as lora strategy or not')
training_parser.add_argument('--which_layer', type=int, default=-1, help='The number of the last layer, whose hidden representation as the embedding')
training_parser.add_argument('--batch_size_per_gpu', type=int, default=300, help='The batch size in per GPU when DDP training')
training_parser.add_argument('--num_epochs', type=int, default=1, help='Training epoch number')
training_parser.add_argument('--max_length', type=int, default=512, help='The max token lenght of the training text')
training_parser.add_argument('--task_prompt', action='store_true', default=False, help='Using task prompt or not')
training_parser.add_argument('--checkpoint_batch_size', type=int, default=10, help='The batch size in checkpointing training strategy')
training_parser.add_argument('--gradcache_chunk_size', type=int, default=10, help='The chunk size in GradCache training strategy')
training_parser.add_argument('--ckpt_saving_dir', type=str, default='/fs-computility/llm/chenzhi/ckpts', help='The saving path of the embedder checkpoint')
training_parser.add_argument('--save_ckpt_steps', type=int, default=500, help='The saving steps of the embedder checkpoint')
training_parser.add_argument('--only_last_ckpt', action='store_true', default=False, help='Only saving the last checkpoint or not')
training_parser.add_argument('--hard_negative_sampling', action='store_true', default=False, help='The contrastive loss function is hard negative sampling or inbatch negative sampling')
training_parser.add_argument('--hard_negative_num', type=int, default=-1, help='The negative numbers when hard negative sampling is setting')
training_parser.add_argument('--temperature', type=float, default=0.02, help='The temperature of the softmax in the InfoNCE')
training_parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate')
training_parser.add_argument('--warmup_rate', type=float, default=0.01, help='The warmup rate of the learning rate')
training_parser.add_argument('--schedule_type', type=str, default='linear', help='The learning schedule type')
training_parser.add_argument('--embedder_ckpt_path', type=str, default='', help='The evaluated checkpoint of the embedder')
training_parser.add_argument('--mytryoshka_size', type=int, default=4096, help='The selected size in matryoshka representation learning')
training_parser.add_argument('--matryoshka_adaptive_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096], help='The adpative dimensions in matryoshka representation learning')
training_parser.add_argument('--record_log', type=str, default='wandb', help='The record type of the accelerator')
training_parser.add_argument('--wandb_project_name', type=str, default='MistralEmbedder', help='The project name of the init wandb')
training_parser.add_argument('--seed', type=int, default=20, help='Random seed')
training_parser.set_defaults(func=train_embedder)

evaluating_parser =  subparsers.add_parser(name='evaluate', help='evaluating embedder')
evaluating_parser.add_argument('--embedder_name', type=str, default='mistral_embedder', help='The name of the training embedder')
evaluating_parser.add_argument('--backbone_type', type=str, default='Mistral', help='Supported backbone types: [Mistral, BERT]')
evaluating_parser.add_argument('--init_backbone', type=str, default='/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e', help='The parameter path of initial embedder backbone')
evaluating_parser.add_argument('--pool_type', type=str, default='position_weight', help='Supported pool types: [position_weight, mean, eos, cls]')
evaluating_parser.add_argument('--peft_lora', action='store_true', default=False, help='Training as lora strategy or not')
evaluating_parser.add_argument('--which_layer', type=int, default=-1, help='The number of the last layer, whose hidden representation as the embedding')
evaluating_parser.add_argument('--max_length', type=int, default=512, help='The max token lenght of the training text')
evaluating_parser.add_argument('--task_prompt', action='store_true', default=False, help='Using task prompt or not')
evaluating_parser.add_argument('--mytryoshka_size', type=int, default=4096, help='The selected size in matryoshka representation learning')
evaluating_parser.add_argument('--embedding_norm', action='store_true', default=False, help='Normalize the embedding or not')
evaluating_parser.add_argument('--embedder_ckpt_path', type=str, default='', help='The evaluated checkpoint of the embedder')
evaluating_parser.add_argument('--mteb_evaluation_tasks', nargs='+', type=str, default=["Banking77Classification", "EmotionClassification", "ImdbClassification", "MedrxivClusteringS2S", "TwitterSemEval2015", "AskUbuntuDupQuestions", "ArguAna", "NFCorpus", "SciFact", "BIOSSES", "STS17", "STSBenchmark"], help='The evaluation tasks in MTEB')
evaluating_parser.add_argument('--device', type=str, default='cuda', help='loading device')
evaluating_parser.set_defaults(func=evaluate_embedder)

predicting_parser =  subparsers.add_parser(name='predict', help='prediction of embedder')
predicting_parser.add_argument('--embedder_name', type=str, default='mistral_embedder', help='The name of the training embedder')
predicting_parser.add_argument('--backbone_type', type=str, default='Mistral', help='Supported backbone types: [Mistral, BERT]')
predicting_parser.add_argument('--init_backbone', type=str, default='/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e', help='The parameter path of initial embedder backbone')
predicting_parser.add_argument('--pool_type', type=str, default='position_weight', help='Supported pool types: [position_weight, mean, eos, cls]')
predicting_parser.add_argument('--peft_lora', action='store_true', default=False, help='Training as lora strategy or not')
predicting_parser.add_argument('--which_layer', type=int, default=-1, help='The number of the last layer, whose hidden representation as the embedding')
predicting_parser.add_argument('--max_length', type=int, default=512, help='The max token lenght of the training text')
predicting_parser.add_argument('--task_prompt', action='store_true', default=False, help='Using task prompt or not')
predicting_parser.add_argument('--mytryoshka_size', type=int, default=4096, help='The selected size in matryoshka representation learning')
predicting_parser.add_argument('--embedding_norm', action='store_true', default=False, help='Normalize the embedding or not')
predicting_parser.add_argument('--embedder_ckpt_path', type=str, default='', help='The evaluated checkpoint of the embedder')
predicting_parser.add_argument('--mteb_evaluation_tasks', nargs='+', type=str, default=["Banking77Classification", "AskUbuntuDupQuestions"], help='The evaluation tasks in MTEB')
predicting_parser.add_argument('--device', type=str, default='cuda', help='loading device')
predicting_parser.set_defaults(func=predict_embedder)

args = parser.parse_args()
args.func(args)