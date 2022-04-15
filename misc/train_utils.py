from datasets import load_metric
import torch


def preprocess_tokenize(examples, tokenizer, prefix="", source_lang="de", target_lang="en",):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs,  truncation=True, )
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, )

    # labels["input_ids"] = [
    #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    # ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def get_compute_metrics(tokenizer):
    sacreblue_metric = load_metric('sacrebleu')

    def compute_metrics(eval_pred, tokenizer):

        with torch.no_grad():
            predictions, labels = eval_pred
            predictions = predictions[0].argmax(axis=-1).tolist()
            # Need to map the -100 labels to 0 labels.
            labels[labels == -100] = 0
            # labels = np.expand_dims(labels, axis=0).tolist()
            predictions_text = tokenizer.batch_decode(predictions)

            labels_text = tokenizer.batch_decode(labels)
            # Make sure it has the right dimensions
            labels_text = [[txt] for txt in labels_text]
            result = sacreblue_metric.compute(predictions=predictions_text, references=labels_text)
        return result

    return lambda x: compute_metrics(x, tokenizer)
