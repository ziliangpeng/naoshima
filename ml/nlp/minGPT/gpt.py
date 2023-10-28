"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from loguru import logger

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/wikigpt'
    C.system.gen_len = 2048
    C.system.gen_per_iter = 1000
    C.system.print_per_iter = 10
    C.system.input_file = 'input.txt'
    C.system.resume = 0
    C.system.topk = 3
    C.system.prompt = "Title:"

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 3e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.compile = 0
    C.trainer.data_parallel = 0


    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 512
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(config.system.input_file, 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    if config.system.resume and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        logger.info(f"resumed from {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path + '.bak')

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % config.system.print_per_iter == 0:
            logger.info(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.sum().item():.5f}")

        if trainer.iter_num % config.system.gen_per_iter == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = config.system.prompt
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, config.system.gen_len, temperature=1.0, do_sample=True, top_k=config.system.topk)[0]
                # y = model.generate(x, 1024, temperature=1.0, do_sample=False, top_k=1)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            with open(os.path.join(config.system.work_dir, f"completion-{trainer.iter_num}.txt"), "w") as f:
                f.write(completion)
            print("done saving model")
            # ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    """
    batch size: 32, block size: 512, gopher, almost max out V100 memory.

    python gpt.py --model.model_type=gpt2 --trainer.compile=?  --trainer.batch_size=8 --data.block_size=512 --system.gen_len=5000 --system.print_per_iter=100
        T4
        compile=~750ms/iter
        non-compile= ~888ms/iter

    python gpt.py --model.model_type=gpt2 --trainer.compile=1  --trainer.batch_size=16 --data.block_size=512 --system.gen_len=5000 --system.print_per_iter=100
        86.46M parameters.
        429ms/iter (V100)
        1394ms/iter (T4) compiled. (not compile will OOM)
        930ms/iter (T4 * 2), no compile

    serious long training:
    python gpt.py --model.model_type=gpt2 --trainer.data_parallel=1 --trainer.compile=0  --trainer.batch_size=26 --data.block_size=512 --system.gen_len=500 --system.print_per_iter=100 --system.resume=1
        number of parameters: 88.81M
        actually using batch size 26 to max out memory.
        16xx ms / iter, T4 * 2.
        450ms/iter, V100 * 2

        V100 * 2, seems to be stuck at 1.6-1.7 loss.
        I messed the model while saving, and have to start again....

        Fuck it, using V100 * 8 and batchsize = 96 in the new attempt.
        about 560-600ms/iter.
        When I do 8 GPU, many GPU have low utilization, sometimes 50%, sometimes 70%, sometimes 100%.
        loss is tablizing at 5.9-6.0 (could improve if I continue to train, but I'm tired of hearing its hallucintion.)
        This is a char level transformer, and it's able to generate words and grammar and sentences that start to be convincing.
        It made up many titles and I had to Google to see if they really exist (most don't)
        This is interesting, because basically my gpt2 already know how to speak like a wikipedia page.


        Next, I am going to train a Chinese language model. I improved the wiki cralwer, to limit to a language, and will prioritize most linked pages.
        This seems to help in narrowing pages to a certain topic related to starting page.
      python gpt.py --model.model_type=gpt2 --trainer.data_parallel=1 --trainer.compile=0  --trainer.batch_size=128 --data.block_size=256 --system.gen_len=1000 --system.print_per_iter=100 --system.resume=0

        I used a small input, and 256 context length, 128 batch size, and gpt2, and top 3 sampling, it almost remember perfetctly and just repeat the input.
        training loss goes as low as 0.6
        My hypothesis:
          - a lot more information can fit into a 256 Chinese character than 512 English characters.
          - topk=3 instead of 10, limiting its creativity/hallucination.
          - dataset too small, and well under its capacity.
          - so, if I tokenize English better (fit more info in the context length), the result could improve a lot too.


      python gpt.py --model.model_type=gopher --trainer.data_parallel=1 --trainer.compile=0  --trainer.batch_size=32 --data.block_size=128 --system.gen_len=1000 --system.print_per_iter=100 --system.resume=0 --system.topk=10

      python gpt.py --model.model_type=gpt-mini --trainer.data_parallel=0 --trainer.compile=1  --trainer.batch_size=128 --data.block_size=128 --system.gen_len=1000 --system.print_per_iter=100 --system.resume=1 --system.prompt=人类的未来


        Next time, try smaller batch size (32?) and larger model.

    """

    # run the optimization
    trainer.run()
