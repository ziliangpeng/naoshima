import os
import time
from loguru import logger
import dataloader
import tinygrad.tensor as t
import tinygrad

# @tinygrad.helpers.Context(DEBUG=7)
def main():
    os.environ['CPU'] = '1'
    dataset = "mnist"
    loader = getattr(dataloader, f"load_{dataset}")
    x_train, y_train, x_test, y_test = loader(onehot=True)

    T = tinygrad.helpers.dtypes.float

    x_train = t.Tensor(x_train, dtype=T).reshape(x_train.shape[0], -1)
    x_test = t.Tensor(x_test, dtype=T).reshape(x_test.shape[0], -1)
    y_train = t.Tensor(y_train, dtype=T)
    y_test = t.Tensor(y_test, dtype=T)

    rg = True
    w1 = t.Tensor.normal(x_train.shape[1], 128, requires_grad=rg, dtype=T)
    b1 = t.Tensor.normal(128, requires_grad=rg, dtype=T)
    w2 = t.Tensor.normal(128, 10, requires_grad=rg, dtype=T)
    b2 = t.Tensor.normal(10, requires_grad=rg, dtype=T)

    def predict(x):
        o = x.matmul(w1).relu() + b1
        o = o.matmul(w2) + b2
        o = o.log_softmax()
        return o

    def loss(x, y):
        o = predict(x)
        loss = (o * y).mean().neg()
        return loss

    def correct(y_pred, y):
        return (y_pred.argmax(1) == y.argmax(1)).sum()

    bs = 128
    epoch = 4200
    # best_l = 1e100
    with t.Tensor.train():
        for e in range(epoch):
            logger.info("epoch: %d" % (e))
            start_epoch = time.time()
            for i in range(x_train.shape[0] // bs):
                w1.grad = None
                b1.grad = None
                w2.grad = None
                b2.grad = None

                batch = x_train[i * bs : (i + 1) * bs]
                batch_y = y_train[i * bs : (i + 1) * bs]
                l = loss(batch, batch_y)
                l.backward()
                # if l.numpy() < best_l:
                #     best_l = l.numpy()
                #     logger.info(f"Best loss: {best_l:.2f}")
                # logger.info(l.numpy())
                # logger.info(w1.grad.sum().numpy())
                lr = 1
                w1 = w1 - w1.grad# * lr
                b1 = b1 - b1.grad #* lr
                w2 = w2 - w2.grad #* lr
                b2 = b2 - b2.grad #* lr

            end_epoch = time.time()

            o_test = predict(x_test)
            corrects = correct(o_test, y_test).numpy()
            logger.info(f"corrects: {corrects} / {x_test.shape[0]}. {corrects / x_test.shape[0]:.2f}%")
            logger.info(f"epoch time: {end_epoch - start_epoch:.2f}")

if __name__ == "__main__":
    main()