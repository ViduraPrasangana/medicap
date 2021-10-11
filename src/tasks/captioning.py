# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.captioning_model import IUModel
from tasks.captioning_data import IUDataset, IUTorchDataset, IUEvaluator
from utils import get_device
device = get_device()

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int,args, shuffle=False, drop_last=False) -> DataTuple:
    dset = IUDataset(splits)
    tset = IUTorchDataset(dset,args)
    evaluator = IUEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class IU:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, args = args, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size, args = args,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = IUModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        print("tring to use %s" % device)
        self.model = self.model.to(device)
        print("train using %s" % next(self.model.parameters()).device)
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple,dump = None):
        

        dset, loader, evaluator = train_tuple
        eval_dset, eval_loader, eval_evaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        eval_iter_wrapper = (lambda x: tqdm(x, total=len(eval_loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        train_losses = []
        valid_losses = []
        for epoch in range(args.epochs):
            predictions = {}
            dump_out = {}
            train_loss=0
            valid_loss=0
            for i, (img_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
                # torch.cuda.set_per_process_memory_fraction(0.7)
                self.model.train()
                self.optim.zero_grad()
                caption = [" ".join((["[MASK]"]*(self.model.lxrt_encoder.max_seq_length)))]*len(img_id)

                feats, boxes  = feats.to(device), boxes.to(device)

                prediction = self.model(feats, boxes, caption)

                # assert prediction.dim() == target.dim() == 2
                targets = []
                for (i, tar) in enumerate(target):
                    tokens = self.model.lxrt_encoder.tokenizer.tokenize(tar.strip())
                    ids = self.model.lxrt_encoder.tokenizer.convert_tokens_to_ids(tokens)
                    padding = [0] * (self.model.lxrt_encoder.max_seq_length - len(ids))
                    ids += padding
                    targets.append(ids[:self.model.lxrt_encoder.max_seq_length])
                
                targets = torch.tensor([t for t in targets], dtype=torch.long).to(device)
                loss = self.criterion(prediction.view(-1, self.model.lxrt_encoder.tokenizer.vocab_size()), targets.view(-1))
                # loss = loss * prediction.size(2)

                loss.backward()
                train_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                word_score, word_id = prediction.max(2)
                # print(prediction,word_id)
                for i_id, w_id in zip(img_id, word_id.cpu().numpy()):
                    predictions[i_id] = w_id
                    dump_out[i_id] = " ".join(self.model.lxrt_encoder.tokenizer.convert_ids_to_tokens(w_id))

            if self.valid_tuple is not None:
                for i, (img_id, feats, boxes, sent, target) in eval_iter_wrapper(enumerate(eval_loader)):
                    with torch.no_grad():
                        self.model.eval()
                        caption = [" ".join((["[MASK]"]*(self.model.lxrt_encoder.max_seq_length)))]*len(img_id)

                        feats, boxes  = feats.to(device), boxes.to(device)

                        prediction = self.model(feats, boxes, caption)
                        # assert prediction.dim() == target.dim() == 2
                        targets = []
                        for (i, tar) in enumerate(target):
                            tokens = self.model.lxrt_encoder.tokenizer.tokenize(tar.strip())
                            ids = self.model.lxrt_encoder.tokenizer.convert_tokens_to_ids(tokens)
                            padding = [0] * (self.model.lxrt_encoder.max_seq_length - len(ids))
                            ids += padding
                            targets.append(ids[:self.model.lxrt_encoder.max_seq_length])
                        
                        targets = torch.tensor([t for t in targets], dtype=torch.long).to(device)
                        loss = self.criterion(prediction.view(-1, self.model.lxrt_encoder.tokenizer.vocab_size()), targets.view(-1))
                        # loss = loss * prediction.size(2)

                        valid_loss += loss.item()


            total_train_loss = train_loss/len(loader)
            total_valid_loss = valid_loss/len(eval_loader)
            train_losses.append(total_train_loss)
            valid_losses.append(total_valid_loss)

            if dump is not None:
                dump = dump=os.path.join(args.output, 'train_predict_epo_'+str(epoch)+'.json')
                evaluator.dump_result(dump_out, dump)
            log_str = "\nEpoch %d: Train accuracy %0.2f: Train Loss %0.2f: Validation Loss %0.2f\n" % (epoch, evaluator.evaluate(predictions) * 100.,total_train_loss,total_valid_loss)
            
            # if self.valid_tuple is not None:  # Do Validation
            #     valid_score = self.evaluate(eval_tuple)
            #     if valid_score > best_valid:
            #         best_valid = valid_score
            #         self.save("BEST")

            #     log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
            #                "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')
           
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.plot_diag(train_losses,valid_losses,self.output)
        self.save("LAST")

    def plot_diag(self,train_losses, valid_losses,output):
        plt.plot(train_losses,'-o')
        plt.plot(valid_losses,'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Losses')
        plt.savefig(self.output+'/loss.png')
        plt.show()
    
    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        predictions = {}
        dump_out ={}
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            img_id, feats, boxes, sent = datum_tuple[:4]
            caption = [" ".join((["[MASK]"]*(self.model.lxrt_encoder.max_seq_length)))]*len(img_id)
            with torch.no_grad():
                feats, boxes = feats.to(device), boxes.to(device)
                logit = self.model(feats, boxes, caption)
                score, word_id = logit.max(2)
                for i_id, w_id in zip(img_id, word_id.cpu().numpy()):
                    predictions[i_id] = w_id
                    dump_out[i_id] = " ".join(self.model.lxrt_encoder.tokenizer.convert_ids_to_tokens(w_id))

        if dump is not None:
            evaluator.dump_result(dump_out, dump)
        
        log_str = "\nTesting %0.2f\n" % (evaluator.evaluate(predictions) * 100.)
        print(log_str, end='')
        with open(self.output + "/log.log", 'a') as f:
            f.write(log_str)
            f.flush()

        return predictions

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (img_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(img_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = IU()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=args.batch_size, args=args,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple,dump=os.path.join(args.output, 'train_predict.json'))


