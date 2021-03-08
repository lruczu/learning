class BioBERTException(Exception):
    description = "BioBERTException"

    def __repr__(self):
        return self.description

    def __str__(self):
        return self.description


class InvalidNumberOfTokens(BioBERTException):
    status_code = 400

    def __init__(self, text: str):
        self.description = f"Invalid number of tokens in: {text}"
