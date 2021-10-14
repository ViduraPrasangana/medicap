# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_feat_csv, load_obj_tsv
from lxrt.tokenization import BertTokenizer

#from tasks.vocabulary import Vocabulary


# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 100
FAST_IMG_NUM = 500

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
IU_IMGFEAT_ROOT = 'data/iu_imgfeat/'
SPLIT2NAME = {
    'train': 'train',
    # 'valid': 'val2014',
    # 'minival': 'val2014',
    # 'nominival': 'val2014',
    'test': 'train',
    'dummy':'dummy',
    "testdummy":"testdummy",
}


class IUDataset:
    """
    A IU xray data example in json file:
    {
        "uid": 1213,
        "filename": "1213_IM-0144-2001.dcm.png",
        "projection": "Lateral",
        "MeSH": "Medical Device",
        "Problems": "Medical Device",
        "image": "CHEST, Two (2) Views XXXX, XXXX at XXXX hours.",
        "indication": "Chest pain and weakness.",
        "comparison": "None.",
        "findings": "Frontal and lateral views of the chest with overlying external cardiac monitor leads show normal size and configuration of the cardiac silhouette. Normal pulmonary vasculature and central airways. No focal airspace consolidation or pleural effusion.",
        "impression": "No acute or active cardiac, pulmonary or pleural disease."
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        self.tokenizer = BertTokenizer.from_pretrained(
                    "bert-iu-xray",
                    do_lower_case=True
                )
        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/iu/%s_data.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        unwanted = []
        for i in range(len(self.data)):
            if self.data[i]["findings"] is None:
                unwanted.append(i)
            else:
                self.data[i]["findings_tokens"] = self.tokenizer.tokenize(self.data[i]["findings"].strip())
                self.data[i]["findings_tokens_ids"] = self.tokenizer.convert_tokens_to_ids(self.data[i]["findings_tokens"])
        print("%s datums have null findings" % len(unwanted))
        for j in sorted(unwanted, reverse = True):
            del self.data[j]
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['filename'].split(".")[0]: datum
            for datum in self.data
        }

        # Create vocab
        #self.vocab = Vocabulary(5,annotations=self.data, dataset_type="iu")

    def __len__(self):
        return len(self.data)
    



"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class IUTorchDataset(Dataset):
    def __init__(self, dataset: IUDataset,args):
        super().__init__()
        self.raw_dataset = dataset
        IU_IMGFEAT_ROOT = args.iu_imgfeat_root
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_feat_csv(
                os.path.join(IU_IMGFEAT_ROOT, '%s_feat.csv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            img_id = datum['filename']
            if img_id in self.imgid2img:
                datum["img_id"] = img_id
                self.data.append(datum)
            
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item: int):
        datum = self.data[item]

        item_id = datum['img_id']
        text = datum['findings']

        # Get image info
        img_info = self.imgid2img[item_id]
        # obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            # print(item_id, feats, boxes, text, target)
            return item_id, feats, text, target
        else:
            # print(item_id, feats, boxes, text)
            return item_id, feats, text, text


class IUEvaluator:
    def __init__(self, dataset: IUDataset):
        self.dataset = dataset

    def evaluate(self, predictions: dict):
        image_score = 0
        word_count = 0
        for i_id, pred in predictions.items():
            original_cap_ids = self.dataset.id2datum[i_id]["findings_tokens_ids"]
            pred_as_list = pred.tolist()
            # print(original_cap_ids,pred)
            word_count += len(original_cap_ids)
            for i in original_cap_ids:
                if(i in pred_as_list):
                    image_score +=1
        
        return 0 if(word_count == 0) else image_score / word_count

    def dump_result(self, predictions: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param predictions: dict of image_id --> prediction
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for image_id, prediction in predictions.items():
                result.append({
                    'image_id': image_id,
                    'prediction': prediction
                })
            json.dump(result, f, indent=4, sort_keys=True)


