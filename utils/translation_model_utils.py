import math


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def translate(model, tokenizer, source_texts, batch_size=32, method="ancestral"):
    translations = []
    for lines in batch(source_texts, n=batch_size):
        samples = sample_model(model, tokenizer, lines, method=method)
        decoded_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        translations += decoded_samples
    return translations


def sample_model(model, tokenizer, source_texts, method="ancestral",):
    sample = None

    tokenized = tokenizer(source_texts, return_tensors="pt", padding=True, ).to("cuda")
    if method == "ancestral":
        sample = model.generate(
            **tokenized,
            do_sample=True,
            num_beams=1,
            top_k=0,
        )
    elif method == "beam":
        sample = model.generate(
            **tokenized,
            do_sample=True,
            num_beams=5,
            early_stopping=True
        )

    else:
        raise(ValueError("Method not implemented: {}".format(method)))

    return sample


def batch_sample(model, tokenizer, texts, n_samples=96, batch_size=32):
    samples = []
    n_loops = math.ceil(n_samples/batch_size)

    last_batch_size = n_samples % batch_size
    for i in range(math.ceil(n_samples/batch_size)):
        # Make sure we generate enough samples by dynamic allocating
        n = batch_size
        if i == n_loops- 1:
            n = last_batch_size
            if n == 0:
                n = batch_size
        tokenized = tokenizer(texts, return_tensors="pt", padding=True, ).to("cuda")

        sample = model.generate(
                **tokenized,
                do_sample=True,
                num_beams=1,
                top_k=0,
                num_return_sequences=n

            )
        decoded_samples = tokenizer.batch_decode(sample, skip_special_tokens=True)
        samples += decoded_samples
    return samples