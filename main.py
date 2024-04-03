import os
import warnings
from typing import Dict
from transformers import pipeline, Connversation
from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

def load_model():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    return pipeline("conversational", model=model, tokenizer=tokenizer)

nlp=load_model()
############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    # As it is a simple text model, no configuration is required
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        question=text.strip()
        answer=nlp(question=question, context="", max_length=512)["answer"]
        output.append(answer)

    return SchemaUtil.create(SimpleText(), dict(text=output))
