from comet import load_from_checkpoint, download_model


class CometMetric:

    def __init__(self, model_name="wmt20-comet-da"):
        self.model_name = model_name
        self.model = None

        self.data = []

    def load_model(self):
        if self.model == None:
            model_path = download_model(self.model_name)
            self.model = load_from_checkpoint(model_path)

    def add(self, src, hyp, target):
        self.data.append({"src": src, "mt": hyp, "ref": target})

    def compute(self):
        self.load_model()
        seg_scores, sys_scores = self.model.predict(self.data, batch_size=4, gpus=1)

        return sys_scores
