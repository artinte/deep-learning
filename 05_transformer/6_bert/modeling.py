
import copy


class BertConfig(object):
    """
    Configuration for `BertModel`.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """
        Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings,
            encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
          max_position_embeddings: The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertModel(object):
    """
    BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python

    # Already been converted into WordPiece token ids
    input_ids = torch.tensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.tensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
                                  num_hidden_layers=8, num_attention_heads=6,
                                  intermediate_size=1024)
    model = modeling.BertModel(config=config, is_training=True,
                                input_ids=input_ids, token_type_ids=token_type_ids,
                                input_mask=input_mask)
    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embedding=False,
                 scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_droput_prob = 0.0
