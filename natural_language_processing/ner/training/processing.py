import re


def normalize(text: str) -> str:
    text = re.sub('\([^\s]*?\)', '', text)  # one word in bracket
    text = re.sub('\[[^\s]*?\]', '', text)
    text = re.sub('\{[^\s]*?\}', '', text)

    text = text.strip()
    text = re.sub(' +', ' ', text)

    text = re.sub(' ,', ',', text)
    text = re.sub(' ;', ';', text)
    text = re.sub(' \.', '\.', text)
    text = re.sub(' !', '!', text)
    text = re.sub(' \?', '\?', text)

    text = re.sub(' +', ' ', text)

    return text
