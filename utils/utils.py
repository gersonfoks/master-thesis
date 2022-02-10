import torch.nn.functional as F
import torch

"""
This file contains utilities:
- Calculating the nearest neighbor
- Calculating distances between embeddings

"""


def calculate_simularity(hypothesis_embeddings, refs_embeddings, measure="cos"):
    hypothesis_repeated = hypothesis_embeddings.repeat(refs_embeddings.shape[0],
                                                       1)  # Make sure there are as many hypothesis ar references
    similarity = None
    if measure == "cos":
        sim_function = torch.nn.CosineSimilarity(dim=-1)
        similarity = sim_function(hypothesis_repeated, refs_embeddings)
    elif measure == "euclidean":
        raise NotImplementedError()
    else:
        raise ValueError("not a known measure: {}".format(measure))

    return similarity


def nearest_neigbhor(vector, references, measure="cos"):
    if measure == "cos":
        sim_function = torch.nn.CosineSimilarity(dim=-1)
        similarity = sim_function(vector.squeeze(dim=0), references.squeeze(dim=0))
    elif measure == "euclidean":
        similarity = -torch.cdist(vector, references, p=2)
    # Need most similar

    index = torch.argmax(similarity)
    return index


def get_distances(vectors_1, vectors_2, measure="cos"):
    distances = []

    for v_1, v_2 in zip(vectors_1, vectors_2):
        distances.append(get_distance(v_1, v_2, measure))
    return distances


def get_distance(vector_1, vector_2, measure="cos"):
    if measure == "cos":
        sim_function = torch.nn.CosineSimilarity(dim=-1)
    elif measure == "euclidean":
        sim_function = lambda v1, v2: -torch.cdist(v1, v2, p=2)
    simularity = sim_function(vector_1, vector_2)

    return simularity.item()


def get_all_word_embeddings(word_embeddings, device="cuda"):
    """
    Get all the word embeddings
    :param word_embeddings:
    :param device:
    :return:
    """

    num_embeddings = word_embeddings.num_embeddings

    numbers = torch.arange(0, num_embeddings - 1).reshape(1, -1).to(device)

    embeddings = word_embeddings(numbers)

    return embeddings


def get_token_embedding(token, word_embeddings, device="cuda"):
    v = torch.Tensor([token]).long().to(device)
    word_embeddings = word_embeddings
    v_embed = word_embeddings(v)

    return v_embed


def embedding_to_indices(embeddings, references):
    r = []

    for embedding in embeddings:
        r.append(nearest_neigbhor(embedding.unsqueeze(0), references))
    return r


def embedding_to_sentence(embeddings, references, tokenizer):
    '''
    Maps the word embedding to a sentence with help of nearest neighbours
    '''
    indices = embedding_to_indices(embeddings[0], references)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(indices))


def ids_to_sentence(ids, tokenizer):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))


def get_word_embedding(tokenizer, word_embeddings, word, device="cuda"):
    tokens = tokenizer.encode(word)

    token = tokens[1]

    embedding = get_token_embedding(token, word_embeddings, device=device)
    return embedding


def find_closest_embedding(tokenizer, word_embeddings, embedding, top=10, measure="cos", device="cuda"):
    all_embeddings = get_all_word_embeddings(word_embeddings, device=device).squeeze(dim=0)

    distances = []

    for embedding_2 in all_embeddings:
        distances.append(get_distance(embedding, embedding_2.unsqueeze(dim=0), measure=measure))

    tokens_distances = [(t, d) for t, d in enumerate(distances)]

    tokens_distances.sort(key=lambda x: x[1], reverse=True)

    new_tokens = [[token] for token, distances in tokens_distances[:top]]
    words = tokenizer.batch_decode(new_tokens)

    result = [{"word": word, "token": token, "score": score}
              for word, (token, score) in zip(words, tokens_distances[:top])
              ]

    return result


def find_closest_words(tokenizer, word_embeddings, word, top=10, measure="cos", device="cuda"):
    tokens = tokenizer.encode(word)

    token = tokens[1]

    embedding = get_token_embedding(token, word_embeddings, device=device)

    return find_closest_embedding(tokenizer, word_embeddings, embedding, top=top, measure=measure, device=device)


def analogous_reasoning(a, b, c, tokenizer, word_embeddings, measure="cos", device="cuda", top=10):
    '''
    a is to b as c is to d
    '''
    a_embedding = get_word_embedding(tokenizer, word_embeddings, a, device=device)
    b_embedding = get_word_embedding(tokenizer, word_embeddings, b, device=device)
    c_embedding = get_word_embedding(tokenizer, word_embeddings, c, device=device)

    d_embedding = b_embedding - a_embedding + c_embedding

    return find_closest_embedding(tokenizer, word_embeddings, d_embedding, top=top, measure=measure, device=device)


def get_distance_between_words(word_1, word_2, tokenizer, word_embeddings, measure="cos", device="cuda"):
    embedding_1 = get_word_embedding(tokenizer, word_embeddings, word_1, device=device)
    embedding_2 = get_word_embedding(tokenizer, word_embeddings, word_2, device=device)

    return get_distance(embedding_1, embedding_2, measure=measure)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings
