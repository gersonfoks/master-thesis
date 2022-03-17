from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.pools_utils import max_pooling, average_pooling

# A wrapper for comet (makes predicting faster then when using as described in the documentation of comet.
from utils.translation_model_utils import batch


class CometWrapper:

    def __init__(self, cometModel, device='cuda'):
        super().__init__()

        self.model = cometModel

        self.device = device

    def predict(self, samples):
        prepared_samples = self.model.prepare_for_inference(samples, )
        inputs = {k: v.to(self.device) for k, v in prepared_samples.items()}
        return self.model.forward(**inputs)

    def fast_predict(self, source, hypothesis, refs):
        src_inputs = self.model.encoder.prepare_sample([source]).to(self.device)
        hyp_inputs = self.model.encoder.prepare_sample(hypothesis).to(self.device)
        ref_inputs = self.model.encoder.prepare_sample(refs).to(self.device)

        n_refs = len(refs)

        src_sent_embed = self.model.get_sentence_embedding(**src_inputs)
        hyp_sent_embed = self.model.get_sentence_embedding(**hyp_inputs)
        ref_sent_embed = self.model.get_sentence_embedding(**ref_inputs)
        # Get the embedding of the hypothesis
        # hyp_sent_emb = self.model.compute_sentence_embedding(mt_input_ids, mt_attention_mask)
        # Get the embedding of the references

        src_sent_embed = src_sent_embed.repeat(n_refs, 1)

        scores = []
        for h in hyp_sent_embed:
            h = h.unsqueeze(dim=0).repeat(n_refs, 1)
            diff_ref = torch.abs(h - ref_sent_embed)
            diff_src = torch.abs(h - src_sent_embed)

            prod_ref = h * ref_sent_embed
            prod_src = h * src_sent_embed

            embedded_sequences = torch.cat(
                (h, ref_sent_embed, prod_ref, diff_ref, prod_src, diff_src),
                dim=1, )

            scores.append(self.model.estimator(embedded_sequences).cpu().numpy().flatten())

        return scores

    def fast_predict_batched(self, source, hypothesis, references, hyp_batch_size=50, ref_batch_size=500):

        scores = [[] for i in range(len(hypothesis))]
        src_inputs = self.model.encoder.prepare_sample([source]).to(self.device)
        src_sent_embed = self.model.get_sentence_embedding(**src_inputs)
        for refs in batch(references, n=ref_batch_size):

            ref_inputs = self.model.encoder.prepare_sample(refs).to(self.device)

            n_refs = len(refs)

            src_sent_embed_repeated = src_sent_embed.repeat(len(refs), 1)

            ref_sent_embed = self.model.get_sentence_embedding(**ref_inputs)

            for hyp in batch(hypothesis, n=hyp_batch_size):

                hyp_inputs = self.model.encoder.prepare_sample(hyp).to(self.device)
                hyp_sent_embed = self.model.get_sentence_embedding(**hyp_inputs)

                for i, h in enumerate(hyp_sent_embed):
                    h = h.unsqueeze(dim=0).repeat(n_refs, 1)
                    diff_ref = torch.abs(h - ref_sent_embed)
                    diff_src = torch.abs(h - src_sent_embed_repeated)

                    prod_ref = h * ref_sent_embed
                    prod_src = h * src_sent_embed_repeated

                    embedded_sequences = torch.cat(
                        (h, ref_sent_embed, prod_ref, diff_ref, prod_src, diff_src),
                        dim=1, )

                    scores[i] += list(self.model.estimator(embedded_sequences).cpu().numpy().flatten())

        return scores
