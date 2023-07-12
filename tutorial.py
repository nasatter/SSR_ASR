from utils import generate_dict, custom_dataset, gen_vocab, gen_audio_sample, compute_metrics, prepare_dataset
from torch.utils.data import Dataset,DataLoader
from transformers import TrainingArguments
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer
from datasets import Dataset, DatasetDict, IterableDataset
import numpy as np
import random
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding():
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



def create_vocab(custom_set):
    # set data_loader for processing vocab
    vocab_dict = gen_vocab(data_loader)
    print(vocab_dict)
    return custom_set


if __name__ == '__main__':
    # First log the dataset and generate training and testing dictionaries
    Gset = generate_dict('C:\\projects\\Phoneme_seg\\val_data\\')
    dataset_dict = Gset.generate()

    # Place data into dataloader for batch processing
    custom_set = custom_dataset(dataset_dict)
    data_loader = DataLoader(custom_set, batch_size=5, shuffle=True, num_workers=5,pin_memory=True,persistent_workers=True)

    dataset_iter = Dataset.from_generator(data_loader)
    dataset_train_test = DataLoader(dataset_iter.with_format("torch"))
    dataset = dataset_train_test.train_test_split(test_size=0.2)
    #create_vocab(data_loader)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=1000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    rand_int = random.randint(0, len(dataset["train"]))-1
    smp = np.array(dataset["train"][rand_int]['audio']['array'])
    rate = dataset["train"][rand_int]['audio']['sampling_rate']
    gen_audio_sample(smp, rate=rate)
    dataprep = prepare_dataset(processor)


    dataset = dataset.map(dataprep.prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    ).to('cuda')

    training_args = TrainingArguments(
    output_dir='C:\\projects\\ASR',
    group_by_length=True,
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True, 
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    )

    trainer.train()


