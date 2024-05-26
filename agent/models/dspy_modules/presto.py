import dspy

class Query(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought('question -> answer')
        