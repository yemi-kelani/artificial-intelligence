import re
import json

class ConceptParser:
    """
    ConceptParser converts a list of triples into readable sentences.
    When it parses a set of concepts it creates a vocabulary 
    containing important tokens from the previous parsing session.
    These can be used to weight the branches in neurologic decoding
    but they generally should not serve as contraints.
    """
    
    def __init__(self, stopwords_path:str = ""):
        self.vocab = None
        self.stopwords = []
        
        if stopwords_path != "":
            with open(stopwords_path) as file:
                self.stopwords = json.load(file)["stopwords"]["english"]
                # print("stopwords:", self.stopwords)
    
    def relation2sentence(self, node:dict) -> str:
        relation = re.split('(?<=.)(?=[A-Z])', node["relationship"])
        
        if relation[0].lower() != "is":
            relation.insert(0, "is")
            
        relation.insert(0, node["start"])
        relation.append(node["end"])
        
        text = " ".join(relation) + "."
        return text.lower()

    def concepts2paragraph(
        self, concept_set:list, 
        set_vocab:bool = True,
        return_list:bool = False
        ) -> str or list:
        
        text = ""
        for concept in concept_set:
            text += " " + self.relation2sentence(concept)
            
        if set_vocab:
            split_text = re.split("[\W]+", text)
            unique_tokens = list(set(split_text))
            
            self.set_vocab( 
                sorted([
                    token for token in unique_tokens 
                    if token not in self.stopwords
                ])
            )
        
        if return_list:
            return split_text[1:-1]
        
        return text
    
    def get_vocab(self):
        return self.vocab
    
    def set_vocab(self, vocab:list) -> None:
        if not isinstance(vocab, list):
            raise ValueError(
                f"Parser vocabulary must be a list, not {type(vocab)}."
            )
        
        del self.vocab
        self.vocab = vocab
        return self
    
    def score(self, text:str) -> float:
        if self.vocab == None or self.vocab == []:
            raise ValueError("Parser vocabulary has not been set.")
        
        candidates = list(set(re.split("[\W]+", text)))
        
        count = 0
        for token in candidates:
            if token in self.vocab:
                count += 1
        
        return count / len(self.vocab)
    
# cs = [
#     {'start': 'a carpet', 'end': 'a house', 'relationship': 'AtLocation', 'weight': 7.745966692414834},
#     {'start': 'the carpet pad', 'end': 'the carpet', 'relationship': 'AtLocation', 'weight': 6.0},
#     {'start': 'floor', 'end': 'carpet', 'relationship': 'RelatedTo', 'weight': 5.7019294979857476},
#     {'start': 'the floor', 'end': 'the carpet', 'relationship': 'AtLocation', 'weight': 5.656854249492381},
#     {'start': 'the pad', 'end': 'the carpet', 'relationship': 'AtLocation', 'weight': 4.0},
#     {'start': 'a carpet', 'end': 'walking on', 'relationship': 'UsedFor', 'weight': 4.0},
#     {'start': 'dust', 'end': 'the carpet', 'relationship': 'AtLocation', 'weight': 3.4641016151377544},
#     {'start': 'carpet', 'end': 'a desk', 'relationship': 'AtLocation', 'weight': 2.82842712474619},
#     {'start': 'a carpet', 'end': 'a bedroom', 'relationship': 'AtLocation', 'weight': 2.82842712474619},
#     {'start': 'Something you find inside', 'end': 'a carpet', 'relationship': 'IsA', 'weight': 2.82842712474619},
#     {'start': 'carpet', 'end': 'floor', 'relationship': 'RelatedTo', 'weight': 2.31862027939031},
#     {'start': 'carpet', 'end': 'rug', 'relationship': 'RelatedTo', 'weight': 2.1033306920215846}
# ]
# parser = ConceptParser("./utilities.json")
# text = parser.concepts2paragraph(cs)
# test =  "I think you'll find a carpet in the bedroom behind the desk. If not in that location, check the pad."
# print(parser.score(" ".join(parser.get_vocab())), parser.score(""))
# print(parser.concepts2paragraph(cs, return_list=True))
# print(parser.concepts2paragraph(cs))