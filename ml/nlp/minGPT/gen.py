import pickle
import torch

from loguru import logger

def gen():
    # Load the entire model
    device = 'mps'
    gen_len = 4

    model = torch.load('wiki-gpt.pt')
    model = model.to(device)

    with open('train_dataset_stoi.pickle', 'rb') as f:
        stoi = pickle.load(f)
    logger.info("Loaded from pickle")

    itos = {v: k for k, v in stoi.items()}
    logger.info("reconstructed itos")

    # Make sure to call model.eval() if you are doing inference only
    model.eval()

    # Now you can perform a forward pass with the model
    # with torch.no_grad():  # Use torch.no_grad() if you're only doing inference
    #     output = model(input_data)

    with torch.no_grad():
        # sample from the model...
        # context = config.system.prompt
        context = "你"
        x = torch.tensor([stoi[s] for s in context], dtype=torch.long)[None,...].to(device)
        topk = 5
        y = model.generate(x, gen_len, temperature=1.0, do_sample=True, top_k=topk)[0]
        # y = model.generate(x, 1024, temperature=1.0, do_sample=False, top_k=1)[0]
        completion = ''.join([itos[int(i)] for i in y])
        logger.info(completion)

if __name__ == '__main__':
    gen()