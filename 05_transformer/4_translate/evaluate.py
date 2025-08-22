import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
