from rouge_score import rouge_scorer

class ROUGE:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def get_RougeL(self, string_1: str, string_2: str):
        score = self.scorer.score(string_1, string_2)

        return score["rougeL"].fmeasure

    def get_Rouge1(self, string_1: str, string_2: str):
        score = self.scorer.score(string_1, string_2)

        return score["rouge1"].fmeasure

    def get_Rouge2(self, string_1: str, string_2: str):
        score = self.scorer.score(string_1, string_2)

        return score["rouge2"].fmeasure