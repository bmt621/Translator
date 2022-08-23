from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import numpy as np


class Tokenizer:
    def __init__(self,tokenizer_path:None):
        """If tokenizer_path is none and tokenizer name is None, 
           we directly download the tokenizer from t5-small
           tokenizer
        """

        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        
        else:
            self.tokenizer = self.load_tokenizer_from_path(tokenizer_path)

        
    def load_tokenizer(self,path):
        AutoTokenizer.from_pretrained(path)

    def save_tokenizer_to_path(self,path):
        AutoTokenizer.save_pretrained(path)

    
    def process_function(self,data):
        inputs = [self.prefix + example[self.src_lang] for example in data["translation"]]
        targets = [example[self.trg_lang] for example in data["translation"]]
        model_inputs = self.tokenizer(inputs, max_length=400, truncation=True)

        with self.tokenizer.as_target_tokenizer():

             labels = self.tokenizer(targets, max_length=400,truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def set_prefix(self,prefix):
        self.prefix = prefix
    
    def set_lang(self,src_lang,trg_lang):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
    
    

class T5_Model(Tokenizer):
    def __init__(self,model_path=None,tokenizer_path = None):
        """If model_path is none and tokenizer name is None, 
           we directly download the tokenizer from t5-small
           model
        """
        super().__init__(tokenizer_path)
        if model_path is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

        else:
            self.model = self.load_from_path(model_path)
        


    def load_from_path(self,model_path):
        return AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def set_trainer(self,dataset,output_dir,epochs:int=10,strategy: str = 'epoch', lr: float = 2e-5,
                    train_batch_size:int=16, decay: float = 0.01,limit:int=3):

        new_dataset = dataset.map(self.process_function,batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            evaluation_strategy=strategy,
            learning_rate = lr,
            per_device_train_batch_size=train_batch_size,
            weight_decay=decay,
            save_total_limit=limit,
            num_train_epochs=epochs
        )
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        self.trainer = Seq2SeqTrainer(model=self.model,
                                 args=training_args,
                                 train_dataset=new_dataset["train"],
                                 tokenizer=self.tokenizer,
                                 data_collator=data_collator,
                               )

    def train(self):
        self.trainer.train()

    
    def translate(self,sentence,num_beams:int=1):

        new_sentence = self.prefix+sentence

        tokenized_sentence = self.tokenizer(new_sentence,max_length=400,truncation=True,return_tensors='pt').input_ids
        out = self.model.generate(tokenized_sentence,num_beams = num_beams)
        translated = self.tokenizer.decode(out[0],skip_special_tokens=True)
  
        return translated