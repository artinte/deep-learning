
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
               msx_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
      """
      Constructs BertConfig.

      Args:
      
      """


class BertModel(object):
  """

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
        
