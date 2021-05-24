from functools import reduce
import glob
import json
import os
from typing import Dict, List

import en_core_web_sm
import jsonlines

from src.ner.common.utils import clean_text

nlp = en_core_web_sm.load()
nlp.max_length = 50000000


def split_text_into_sentences(text) -> List[str]:
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


class Section:
    def __init__(self, title: str, content: str):
        self.title = title
        self.content = content

    def __repr__(self):
        return f'Title: {self.title}\n\nContent: {self.content}\n'

    def __str__(self):
        return f'{self.title} {self.content}'

    def to_dict(self):
        return {
            'Title': self.title,
            'Content': self.content,
        }


class Document:
    def __init__(
        self,
        document_id: str,
        sections: List[Section],
    ):
        """
        Args:
            document_id: str
            sections:
        """
        self.document_id = document_id
        self.sections = sections
        self._labels = set()
        self.mapping: Dict[str, List[str]] = dict()  # label -> sentences with it

    @classmethod
    def read_from_path(cls, json_path: str) -> "Document":
        id_ = os.path.basename(json_path).replace('.json', '')
        sections = []
        with open(json_path, 'r') as f:
            json_decode = json.load(f)
            for data in json_decode:
                section = Section(data['section_title'], data['text'])
                sections.append(section)
        return cls(id_, sections)

    def get_sentences_with_label(self, label: str, fast_process: bool = True) -> List[str]:
        label_ = clean_text(label)
        doc_ = clean_text(self.__str__())
        if label_ not in doc_:
            raise ValueError(f'Label: {label} could not be found in dataset')

        sentences_with_label = []
        if fast_process:
            for section in self.sections:
                section_doc = section.__str__()
                if label_ in clean_text(section_doc):
                    for sentence in split_text_into_sentences(section_doc):
                        if label_ in clean_text(sentence):
                            sentences_with_label.append(sentence)

            return sentences_with_label

        for sentence in split_text_into_sentences(self.__str__()):
            if label_ in clean_text(sentence):
                sentences_with_label.append(sentence)

        return sentences_with_label

    def get_all_sentences(self) -> List[str]:
        sentences = []
        for section in self.sections:
            sentences.append(split_text_into_sentences(section.content))
        sentences = reduce(lambda a, b: a + b, sentences)
        return sentences

    def add_labels(self, labels: List[str]):
        self._labels = set(labels)

    def load_sentences_to_mapping(self):
        for label in self._labels:
            self.mapping[label] = self.get_sentences_with_label(label)

    def to_dict(self):
        return {
            'document_id': self.document_id,
            'sections': [s.to_dict() for s in self.sections],
            'mapping': self.mapping,
        }

    def __str__(self):
        string = ''
        for section in self.sections:
            string += section.__str__() + ' '

        return string.strip()


class Dataset:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    @classmethod
    def read_from_directory(cls, directory_pattern: str) -> "Dataset":
        documents = []
        for path in glob.glob(directory_pattern):
            documents.append(Document.read_from_path(path))
        return cls(documents)

    def get_by_id(self, id_: str) -> Document:
        for document in self.documents:
            if document.document_id == id_:
                return document
        raise ValueError('Provided: {} does not exist in dataset')

    def list(self) -> List[str]:
        return [d.__str__() for d in self.documents]

    def save(self, json_path):
        with jsonlines.open(json_path, 'w') as writer:
            for document in self:
                writer.write(document.to_dict())

    @classmethod
    def load(cls, json_path) -> "Dataset":
        documents = []

        with jsonlines.open(json_path) as reader:
            for line in reader:
                doc = Document(
                    document_id=line['document_id'],
                    sections=line['sections']
                )
                doc.labels = set(line['mapping'])
                doc.mapping = line['mapping']
                documents.append(doc)
        return cls(documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]
