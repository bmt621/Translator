from pidgin_model import T5_Model
from data_process import DataProcessor
from torchtext.utils import download_from_url, extract_archive
import io


"""
                              INFERENCE

model = T5_Model()

---------------------------------set prefix------------------------------------
prefix = 'Translate English to German: '
model.set_prefix(prefix)

---------------------------------set src_lang and trg_lang----------------------
src_lang = 'en'
trg_lang = 'ger'

model.set_lang(src_lang,trg_lang)

---------------------------------translate sentence-----------------------------
sentence = 'hello, how are you'
translated = model.translate(sentence)

                             TRAINIING MODEL


url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')


train_file_paths = [extract_archive(download_from_url(url_base+url))[0] for url in train_urls]
val_file_paths = [extract_archive(download_from_url(url_base+url))[0] for url in val_urls]
test_file_paths = [extract_archive(download_from_url(url_base+url))[0] for url in test_urls]

def extract_sentences(filepath):

  
  german_iters = iter(io.open(filepath[0],encoding='utf8'))
  english_iters = iter(io.open(filepath[1],encoding='utf8'))

  ger_txt, eng_txt = [],[]
  
  for german_iter,english_iter in zip(german_iters,english_iters):
    german_text = german_iter.rstrip("\n")
    english_text = english_iter.rstrip("\n")

    ger_txt.append(german_text)
    eng_txt.append(english_text)
    

  return ger_txt,eng_txt



train_text= extract_sentences(train_file_paths)

data_processor = DataProcessor('example.json')

new_data = data_processor.__output__(train_text[0],train_text[1])

print(new_data)

model = T5_Model()
model.set_prefix('Tranlate English to German: ')
model.set_lang("en","ger")

model.set_trainer(new_data,output_dir="./eng_ger_model")
model.train()

"""