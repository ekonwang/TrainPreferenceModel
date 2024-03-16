dataset_sampling_ratios = {
    'ELI5': 1.0,
    'HotpotQA': 1.0,
    'MSMARCO': 1.0,
    'MultiNLI': 1.0,
    'Quora': 1.0,
    'MIRACL': 1.0,
    'MrTyDi': 1.0,
    'SQuAD': 1.0,
    'NautralQuestions': 1.0,
    'TriviaQA': 1.0,
    'FEVER': 1.0,
    'DuReader': 1.0,
    'T2Ranking': 1.0,
    'MSMARCO_Triple': 1.0,
    'STAllNLI': 1.0,
    'STELI5': 1.0,
    'STGooQA': 1.0,
    'STSpecter': 1.0,
    'STStackexchangeDup': 1.0,
    'STWikiHow': 1.0,
    'STYahooQA': 1.0,
    'STWikiAnswers': 1.0,
    'STAGNews': 1.0,
    'STAltlex': 1.0,
    'STAmazonReview': 1.0,
    'STCodeSearchNet': 1.0,
    'STFlickr30k': 1.0,
    'STNPR': 1.0,
    'STPAQ': 1.0,
    'STS2ORCTA': 1.0,
    'STXSum': 1.0,
    'STCCNews': 1.0,
    'MTWoW': 1.0,
    'MTTrex': 1.0,
    'MTMedMCQA': 1.0,
    'MTPubMed': 1.0,
    'NOMICTriples': 1.0,
    'GAOKAO': 1.0,
    'SFREmotion': 1.0,
    'SFRFiQA': 1.0,
    'SFRMTOPIntent': 1.0,
    'SFRSTS12': 1.0,
    'SFRSTS22': 1.0,
    'SFRSTSBenchmark': 1.0,
    'SFRToxicConversation': 1.0,
    'SFRTweetSentiment': 1.0,
    'SFRbioRxiv': 1.0,
    'SFRmedRxiv': 1.0,
    'SFRSciDocs': 1.0
}

# dataset_sampling_ratios = {
#     'ELI5': 1.0,
#     'HotpotQA': 1.0,
#     'MSMARCO': 0.922,
#     'MultiNLI': 1.0,
#     'Quora': 1.0,
#     'MIRACL': 1.0,
#     'MrTyDi': 1.0,
#     'SQuAD': 1.0,
#     'NautralQuestions': 1.0,
#     'TriviaQA': 1.0,
#     'FEVER': 1.0,
#     'DuReader': 1.0,
#     'T2Ranking': 1.0,
#     'MSMARCO_Triple': 1.0,
#     'STAllNLI': 1.0,
#     'STELI5': 1.0,
#     'STGooQA': 0.520,
#     'STSpecter': 1.0,
#     'STStackexchangeDup': 1.0,
#     'STWikiHow': 1.0,
#     'STYahooQA': 1.0,
#     'STWikiAnswers': 0.535,
#     'STAGNews': 0.969,
#     'STAltlex': 1.0,
#     'STAmazonReview': 0.658,
#     'STCodeSearchNet': 0.291,
#     'STFlickr30k': 1.0,
#     'STNPR': 0.983,
#     'STPAQ': 0.572,
#     'STS2ORCTA': 0.201,
#     'STXSum': 1.0,
#     'STCCNews': 0.912,
#     'MTWoW': 1.0,
#     'MTTrex': 0.327,
#     'MTMedMCQA': 1.0,
#     'MTPubMed': 0.734,
#     'NOMICTriples': 1.0,
#     'GAOKAO': 1.0,
#     'SFREmotion': 1.0,
#     'SFRFiQA': 1.0,
#     'SFRMTOPIntent': 1.0,
#     'SFRSTS12': 1.0,
#     'SFRSTS22': 1.0,
#     'SFRSTSBenchmark': 1.0,
#     'SFRToxicConversation': 1.0,
#     'SFRTweetSentiment': 1.0,
#     'SFRbioRxiv': 1.0,
#     'SFRmedRxiv': 1.0
# }

dataset_task_prompts = {
    'ELI5': [
        'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum'
    ],
    'HotpotQA': [
        'Given a multi-hop question, retrieve documents that can help answer the question'
    ],
    'MSMARCO': [
        'Given a web search query, retrieve relevant passages that answer the query',
        'Given a web search query, retrieve relevant documents that answer the query'
    ],
    'MultiNLI': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'Quora': [
        'Given a question, retrieve questions that are semantically equivalent to the given question',
        'Find questions that have the same meaning as the input question'
    ],
    'MIRACL': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'MrTyDi': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'SQuAD': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'NautralQuestions': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'TriviaQA': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'FEVER': [
        'Given a claim, retrieve documents that support or refute the claim'
    ],
    'DuReader': [
        'Given a Chinese search query, retrieve web passages that answer the question'
    ],
    'T2Ranking': [
        'Given a Chinese search query, retrieve web passages that answer the question'
    ],
    'MSMARCO_Triple': [
        'Given a web search query, retrieve relevant passages that answer the query',
        'Given a web search query, retrieve relevant documents that answer the query'
    ],
    'STELI5': [
        'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum'
    ],
    'STAllNLI': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'STGooQA': [
        'Provided a google question, retrieve the highest voted answers'
    ],
    'STSpecter': [
        'Provided a title of the scientific publication, retrieve the related title of the publication'
    ],
    'STStackexchangeDup': [
        'Given a question, retrieve questions that are semantically equivalent to the given question',
        'Find questions that have the same meaning as the input question'
    ],
    'STWikiHow': [
        'Given a summary, retrieve the corresponding documents'
    ],
    'STYahooQA': [
        'Provided a Yahoo question, retrieve the highest voted answers'
    ],
    'STWikiAnswers': [
        'Retrieve wikipedia query that are semantically similar to the given query'
    ],
    'STAGNews': [
        'Given an AGNews title, retrieve the corresponding news description'
    ],
    'STAltlex': [
        'Given a wikipedia passage, retrieve the simplified version'
    ],
    'STAmazonReview': [
        'Given an Amazon review title, retrieve the corresponding review content'
    ],
    'STCodeSearchNet': [
        'Given a code comment, retrieve the corresponding code'
    ],
    'STFlickr30k': [
        'Find image captions that have the same meaning as the input caption'
    ],
    'STNPR': [
        'Given an Pushshift title, retrieve the corresponding Pushshift body'
    ],
    'STPAQ': [
        'Given a question, retrieve web passages that answer the question'
    ],
    'STS2ORCTA': [
        'Given a title of a scientist paper, retrieve the corresponding paper\'s abstract'
    ],
    'STXSum': [
        'Given an news summary, retrieve the corresponding news article'
    ],
    'STCCNews': [
        'Given an news title, retrieve the corresponding news article'
    ],
    'MTWoW': [
        'Given a question, retrieve wikipedia passages that answer the question'
    ],
    'MTTrex': [
        'Given a relation claim, retrieve documents that can extract the relation'
    ],
    'MTMedMCQA': [
        'Given a medical question, retrieve the corresponding answer of the give question'
    ],
    'MTPubMed': [
        'Given a medical question, retrieve passages that answer the question'
    ],
    'NOMICTriples': [
        ''
    ],
    'GAOKAO': [
        ''
    ],
    'SFREmotion': [
        'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise'
    ],
    'SFRFiQA': [
        'Given a financial question, retrieve user replies that best answer the question'
    ],
    'SFRMTOPIntent': [
        'Classify the intent of the given utterance in task-oriented conversation'
    ],
    'SFRSTS12': [
        'Retrieve semantically similar text'
    ],
    'SFRSTS22': [
        'Retrieve semantically similar text'
    ],
    'SFRSTSBenchmark': [
        'Retrieve semantically similar text'
    ],
    'SFRToxicConversation': [
        'Classify the given comments as either toxic or not toxic'
    ],
    'SFRTweetSentiment': [
        'Classify the sentiment of a given tweet as either positive, negative, or neutral'
    ],
    'SFRbioRxiv': [
        'Identify the main category of Biorxiv papers based on the titles and abstracts'
    ],
    'SFRmedRxiv': [
        'Identify the main category of Medrxiv papers based on the titles and abstracts'
    ],
    'SFRSciDocs': [
        'Given a title of a scientific paper, retrieve the titles of other relevant papers'
    ]
}

# initial dataloader
## all samples are randomly sampled, where one batch contains different domain samples. 
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
#                   '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl']

## all samples in a batch are sampled from the same task, which we called the in-domain batch sampling. 
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/InDomain/train.jsonl']

##  msmarco dataset with hard negative sampling
# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/MSMARCO_Triple/train.jsonl']

# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/ELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl'
# ]

# training_datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/STELI5/train.jsonl', 
#                 '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train.jsonl',
#                 '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train.jsonl']


# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/STELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/filtered_phase2_train.jsonl']

# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/STELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STNPR/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STXSum/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/filtered_phase2_train.jsonl']

# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/NOMICTriples/train.jsonl']

# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/STELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/filtered_phase2_train.jsonl']

# training_datatset_files = [
#         '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/Quora/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train_bge_retrieval_triples.jsonl']

# training_datatset_files = [
#         '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/Quora/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STNPR/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STXSum/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/train_bge_retrieval_triples.jsonl']

# training_datatset_files = [
#         '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_biology.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_chemistry.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_history.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_mathematics.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_physics.jsonl'
# ]


# training_datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/STELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFREmotion/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRFiQA/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRMTOPIntent/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS12/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS22/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRSTSBenchmark/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRToxicConversation/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRTweetSentiment/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRbioRxiv/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRmedRxiv/train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SFRSciDocs/dev.jsonl'
# ]

# training_datatset_files = [
#         '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/Quora/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STNPR/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STXSum/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFREmotion/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRFiQA/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRMTOPIntent/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS12/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS22/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRSTSBenchmark/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRToxicConversation/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRTweetSentiment/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRbioRxiv/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRmedRxiv/train.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SFRSciDocs/dev.jsonl'
# ]

training_datatset_files = [
            '/fs-computility/llm/chenzhi/datasets_processed/STELI5/filtered_phase2_train.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/filtered_phase2_train.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STNPR/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STXSum/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/filtered_phase2_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFREmotion/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRFiQA/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRMTOPIntent/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS12/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRSTS22/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRSTSBenchmark/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRToxicConversation/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRTweetSentiment/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRbioRxiv/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRmedRxiv/train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SFRSciDocs/dev.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_biology.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_chemistry.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_history.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_mathematics.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/train_physics.jsonl'
]