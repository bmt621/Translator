from datasets import load_dataset
import json


class DataProcessor:

    def __init__(self,save_to=None):
        self.saved_to = save_to
    
    def write_to_json(self,data_set):

        data_x = data_set[0]
        data_y = data_set[1]

        with open(self.saved_to, 'w') as outfile:
            for id , (ger, eng) in enumerate(zip(data_x,data_y)):
                json_dic = {
                            'id':id,
                            'translation':{'en':eng,'ger':ger}
                        }

                outfile.write(json.dumps(json_dic))

    def __output__(self,x,y):

        self.write_to_json((x,y))
        out_data = load_dataset('json',data_files=self.saved_to)

        return out_data


