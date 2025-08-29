import torch
from positional_encoding import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_head,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        device,
    ):
        """
        Initializes the Transformer model components.

        Args:
            src_vocab_size (int): Size of the source language vocabulary.
            tgt_vocab_size (int): Size of the target language vocabulary.
            d_model (int): The number of expected features in the encoder/decoder inputs (embedding dimension).
            n_head (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
            device (torch.device): The device (e.g., 'cpu' or 'cuda') to run the model on.
        """
        super(TransformerModel, self).__init__()
        self.device = device
        self.d_model = d_model
        self.src_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Set to True for batch-first tensors
        )

        self.generator = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        """
        The forward pass for training the model.

        Args:
            src (torch.Tensor): The source sequence tensor. Shape: [batch_size, src_seq_len]
            tgt (torch.Tensor): The target seqeunce tensor. Shape: [batch_size, tgt_seq_len]
            src_key_padding_mask (torch.Tensor): Boolean mask to prevent attention to padding tokens in the source.
                Shape: [batch_size, src_seq_len]
            tgt_key_padding_mask (torch.Tensor): Boolean mask to prevent attention to padding tokens in the target.
                Shape: [batch_size, tgt_seq_len]

        Returns:
            torch.Tensor: The final output logits for the target sequence. Shape: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        src_emb = self.positional_encoding(self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, device=self.device)))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, device=self.device)))

        # Generate a boolean causal mask for the target sequence
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(self.device)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        # [batch_size, tgt_seq_len, vocab_size]
        return self.generator(out)

    def encode(self, src, src_key_padding_mask):
        """
        Encodes the source sequence into a memory tensor.

        This method is typically used during inference (e.g., beam search)
        when the entire source sequence is encoded once.

        Args:
            src (torch.Tensor): The source sequence tensor. Shape: [batch_size, src_seq_len]
            src_key_padding_mask (torch.Tensor): Boolean mask to prevent attention to padding tokens in the source.
                Shape: [batch_size, src_seq_len]

        Returns:
            torch.Tensor: The output memory tensor from the encoder. Shape: [batch_size, src_seq_len, d_model]
        """
        src_emb = self.positional_encoding(self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, device=self.device)))
        return self.transformer.encoder(
            src=src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, memory_key_padding_mask):
        """
        Decodes the target sequence given the encoder's output memory.

        This method is typically used during inference for step-by-step
        generation of the target sequence.

        Args:
            tgt (torch.Tensor): The current target sequence tensor (may be a single token during decoding).
                Shape: [batch_size, current_tgt_seq_len]
            memory (torch.Tensor): The encoder's output tensor. Shape: [batch_size, src_seq_len, d_model]
            memory_key_padding_mask (torch.Tensor): Boolean mask for the memory (source sequence padding).
                Shape: [batch_size, src_seq_len]

        Returns:
            torch.Tensor: The final output logits for the current target step(s).
                Shape: [batch_size, current_tgt_seq_len, tgt_vocab_size]
        """
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, device=self.device)))

        # Generate a boolean causal mask for the target sequence
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(self.device)

        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.generator(out)
