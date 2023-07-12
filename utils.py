import os, random, json
import numpy as np
from torch.utils.data import Dataset,DataLoader
from scipy.interpolate import interp1d
from scipy.io.wavfile import write

# Creates a dictionary from a directory. The subdirectories in the 
# provided directory specify the classes. The weights are used
# to generate the training, validation, and testing sets, repectively
# If the dictary is not saved, it is recommend to set a seed for
# random. If the data is to be stored in the dictionary, set the 
# store_data flag to true (default). 
class generate_dict():
    def __init__(self,workingdir,store_data = True,weights=[0.7,0.15,0.15]):
        if workingdir[-1] != '\\':
            workingdir+='\\'
        if sum(weights)>1:
            weights = np.array(weights)/sum(weights)
            print('weight sum exceeds unit, using the following weights:'+str(weights))
        self.workingdir = workingdir
        self.store_data = store_data
        self.weights = weights
        self.lower_only = True
        self.file_dict = {}

    def generate(self):
        # Creates classes from folders
        folders = os.listdir(self.workingdir)
        file_dict = {}
        file_dict['file'] = []
        file_dict['text'] = []
        file_dict['class'] = []
        file_dict['audio'] = []
        index = 0
        for classes in folders:
            subdir = self.workingdir+classes+'\\'
            
            if os.path.isdir(subdir):
                files = os.listdir(subdir)
                train_indx = int(len(files)*self.weights[0])
                val_indx = int(len(files)*self.weights[1])+train_indx
                classes = classes.lower() if self.lower_only else None
                random.shuffle(files)
                for i in files:
                    file_dict['file'].append(subdir+i )
                    file_dict['text'].append(classes)
                    file_dict['class'].append(index)
                    if self.store_data:
                        audio_dict = {'array':np.array(np.genfromtxt(subdir+i, delimiter=",").astype(float)).flatten(),'sampling_rate':1000}
                        file_dict['audio'].append(audio_dict)
                index += 1
        print(len(file_dict['file']))
        self.file_dict = file_dict
        return file_dict

# Here we inherit the Dataset properties. This is essentially a generated
# that return a new instance of the data set when called until the number of calls 
# exceeds the len. When passed to the dataloader, the generator is threaded to pass
# batched instances of the data.
class custom_dataset(Dataset):
    def __init__(self, all_data, base_sample_rate=1000, desired_sample_rate=16000):
        self.all_data = all_data
        self.base_sample_rate = base_sample_rate
        self.desired_sample_rate = desired_sample_rate

    def set_dataset(self,selected_set):
        self.current_set = selected_set

    def __len__(self):
        # iterate over the selected set
        return len(self.all_data['text'])
        
    def __getitem__(self, index):
            audio = self.all_data['audio'][index]
            audio = self.upsample(audio)
            text = self.all_data['text'][index]
            clas = self.all_data['class'][index]
            file = self.all_data['file'][index]
            initial_dict = {'file':file, 'text':text, 'class':clas, 'audio':audio}
            final_dict = self.select_random(initial_dict)
            return final_dict

    def upsample(self,data):
        dta_out = []
        for i in range(data.shape[1]):
            x = np.linspace(0, 1, num=len(data[:,i]), endpoint=True)
            y = data[:,i]
            f1 = interp1d(x, y, kind='cubic')
            x1 = f1(np.linspace(0, 1, num=self.desired_sample_rate, endpoint=True))
            dta_out.append(x1)
        dta_out = np.stack(img1,axis=0)
        return dta_out

    def select_random(self, initial_dict):
        num = random.randint(4,10)
        indexes = random.randint([self.__len__()]*num)
        for index in indexes:
            initial_dict['text']+= self.all_data['text'][index]
            audio = self.all_data['audio'][index]
            audio = self.upsample(audio)
            initial_dict['audio'] = np.concatenate((initial_dict['audio'],audio))
        return initial_dict




# This function finds the unique instances of letters in the training set
def gen_vocab(dataset):
    vocab_list = []
    for i,data in enumerate(dataset, 0):
        data_cat = ''.join(data)
        for i in data_cat:
            vocab_list.append(i) if i not in vocab_list else None
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["|"] = len(vocab_dict)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    return vocab_dict

def gen_audio_sample(data,rate,name='test2.wav',intp=False):
    x = np.linspace(0, 1, num=len(data), endpoint=True)
    if intp:
        f1 = interp1d(x, data, kind='cubic')
        x1 = f1(np.linspace(0, 1, num=rate, endpoint=True))
    else:
        x1 = data
    x1 = (x1 - np.min(x1))/(np.max(x1)-np.min(x1))
    scaled = np.int16(x1 / np.max(np.abs(x1)) * 32767)
    write(name, rate, scaled)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


class prepare_dataset():
    def __init__(self,processor):
        self.processor = processor

    def prepare_dataset(self,batch):
        audio = batch["audio"]
    
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch