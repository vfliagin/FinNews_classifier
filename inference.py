import os
from transformers import pipeline

class FinNewsClassifier():
    def __init__(self):
        
        self.path = os.getcwd()
        self.classifier = pipeline("sentiment-analysis", model = self.path + '/saved_model')

    def classify(self, sentences: list):
        
        res = []
        
        for sent in sentences:
            res.append(self.classifier(sent))

        return res


if __name__ == '__main__':
    
    classifier = FinNewsClassifier()
    
    textI = "Shake Shack stock surges 26% on fourth-quarter profit, strong 2024 outlook"
    textII = "Crypto hedge fund accused of ‘criminal’ mismanagement in dispute over FTX"
    textIII = "Norsk Hydro warns of construction demand slump similar to pandemic"

    sentences = [textI, textII, textIII]
    
    print(classifier.classify(sentences))
