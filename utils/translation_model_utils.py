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
