from datasets import load_metric


def get_sacrebleu(tokenizer):
    '''
    Gets the sacreblue metric
    :param tokenizer:
    :return:
    '''
    blue_metric = load_metric('sacrebleu')
    def compute_sacreblue(eval_pred, tokenizer):
        predictions, labels = eval_pred
        predictions = predictions[0].argmax(axis=-1).tolist()
        # Need to map the -100 labels to 0 labels.
        labels[labels == -100] = 0
        #labels = np.expand_dims(labels, axis=0).tolist()
        predictions_text = tokenizer.batch_decode(predictions)

        labels_text = tokenizer.batch_decode(labels)
        # Make sure it has the right dimensions
        labels_text = [[txt] for txt in labels_text]
        result = blue_metric.compute(predictions=predictions_text, references=labels_text)

        return result
    return lambda x: compute_sacreblue(x, tokenizer)