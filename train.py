'''
This script handling the training process.
'''

import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import dataset
from optimizer import ScheduledOptimizer
from models.Functions import BatchToData 


def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(dataset.PADDING_TOKEN).data).sum()

    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        # prepare data
        
        inputs, positions, labels = BatchToData(batch)
        # we assume that both the simple and normal datasets are padded (with ones...)
        
        if next(model.parameters()).is_cuda:
            src = (inputs.cuda(), positions.cuda())
            labels = labels.cuda()
        else:
            src = (inputs, positions)
        
        # forward
        optimizer.zero_grad()
        pred = model(src)

        # backward
        loss = crit(pred, labels)
        # loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total = len(labels)
        n_correct = (pred.max(1)[1].data==targets).long().sum().data[0]
        print(n_correct)
        # n_words = gold.data.ne(dataset.PADDING_TOKEN).sum()
        # n_total_words += n_words
        # n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss / n_total, n_correct / n_total


def eval_epoch(model, validation_data, crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):
        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(dataset.PADDING_TOKEN).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss / n_total_words, n_total_correct / n_total_words


def train(model, training_data, validation_data, crit, optimizer, log='model'):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if log:
        log_train_file = log + '.train.log'
        log_valid_file = log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(100):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch_i}

        model_name = log + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
        torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


def main():
    from torchtext.data.iterator import Iterator
    import torchtext
    from models.Models import AttentiveRelationsNetwork
    
    word_dim = 512
    cuda = True
    batch_size = 64
    max_sentence_len = 240
    n_warmup_steps = 4000

    # ========= Loading Dataset =========#
    training_set, testing_set, vocab = dataset.simple_wikipedia()

    sort_key = lambda batch: torchtext.data.interleave_keys(len(batch.normal), len(batch.simple))
    training_data = Iterator(training_set, batch_size, shuffle=True, device=-1, repeat=False, sort_key=sort_key)
    validation_data = Iterator(testing_set, batch_size, device=-1, train=False, sort_key=sort_key)

    vocab_size = len(vocab)

    # TODO: Incorporate vocab. size as target class length.
    
    model = AttentiveRelationsNetwork(n_src_vocab=vocab_size, n_max_seq=max_sentence_len, 
                                         out_classes=2, batch_size=batch_size)

    optimizer = ScheduledOptimizer(
        optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        word_dim, n_warmup_steps)

    def get_criterion(vocab_size):
        """ With PAD token zero weight """
        weight = torch.ones(vocab_size)
        weight[1] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    # crit = get_criterion(vocab_size)
    crit = nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
        # crit = crit.cuda()

    train(model, training_data, validation_data, crit, optimizer)


if __name__ == '__main__':
    main()
