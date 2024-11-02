# extractors
from summa import keywords
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(self, extractor_type:str = "keybert", limit:int = 5):
        if extractor_type == "textrank":
            self.extractor = keywords
        elif extractor_type == "keybert":
            self.extractor = KeyBERT(model='all-mpnet-base-v2')
        else:
            raise ValueError(f"Keyword Extractor type ' {extractor_type}) ' is unsupported.")
        
        self.type = extractor_type
        self.limit = limit
            
    def extract(self, text: str) -> list:
        if self.type == "textrank":
            extracted_keywords = self.extractor.keywords(text, scores=True)
        elif self.type == "keybert":
            extracted_keywords = self.extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words='english',
                highlight=False,
                top_n=self.limit)

            extracted_keywords = list(dict(extracted_keywords).keys())
        
        return extracted_keywords
    
    def extract_all(self, text_list: list) -> list:
        keyword_list = []
        for text in text_list:
            keyword_list.append(self.extract(text))
        
        return keyword_list         