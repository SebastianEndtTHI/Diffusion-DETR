import torch
from torch import nn
from torch.nn import functional as F


def load_state_dict_compat(model, state_dict):
    """Load a checkpoint while accepting older feature extractor key layouts."""

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    remapped_state_dict = {}
    remapped = False
    for key, value in state_dict.items():
        new_key = key

        if key.startswith("feature_extractor.") and not key.startswith("feature_extractor.encoder."):
            new_key = "feature_extractor.encoder." + key[len("feature_extractor."):]
            remapped = True
        elif key.startswith("feature_extractor.encoder."):
            new_key = "feature_extractor." + key[len("feature_extractor.encoder."):]
            remapped = True

        remapped_state_dict[new_key] = value

    if remapped:
        model.load_state_dict(remapped_state_dict)
        return

    model.load_state_dict(state_dict)



# Defining the MLP encoder that can be used standalone or as feature extractor for the transformer encoder
class MLP_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # network initialization
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.fs_dim = args.fs_dim

        # encoder MLP
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                    nn.LayerNorm(self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.LayerNorm(self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.LayerNorm(self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.fs_dim),
                                    nn.LayerNorm(self.fs_dim),
                                    nn.ReLU())
        
    def forward(self, x):
        hs = self.encoder(x)
        return hs



# Defining the MLP prediction head structure for MD, FA, direction, weight and existence score prediction
class MLP_Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # network initialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        # MLP prediction head
        self.mlp = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.output_dim))
        
    def forward(self, X):
        out = self.mlp(X)
        return out


  
# Defining the core DETR model
class DWI_DETR(nn.Module):
    def __init__(self, args):
        super().__init__()

        # network initialization
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.fs_dim = args.fs_dim
        self.n_queries = args.n_queries
        self.n_layers = args.n_dlayers
        self.n_heads = args.n_multihead

        self.aux_loss = args.aux_loss

        self.combined_mlp_head = args.combined_mlp_head

        self.encoder_type = args.encoder_type



        mlp_encoder = MLP_Encoder(args)

        
        if self.encoder_type == 'MLP':
            # using encoder MLP as feature extractor
            self.encoder = mlp_encoder

        elif self.encoder_type == 'Transformer':
            # using encoder MLP as feature extractor and transformer encoder for contextualized features
            self.feature_extractor = mlp_encoder
            self.encoder_layer = nn.TransformerEncoderLayer(self.fs_dim, 
                                                            self.n_heads, 
                                                            dim_feedforward=self.hidden_dim, 
                                                            batch_first=True)
            self.encoder_norm = nn.LayerNorm(self.fs_dim)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                 num_layers=self.n_layers, 
                                                 norm=self.encoder_norm)
            
            self.encoder = lambda x: self.transformer_encoder(self.feature_extractor(x))
        

        # query initialzation
        self.queries = nn.Embedding(self.n_queries, self.fs_dim)


        # transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(self.fs_dim, 
                                                        self.n_heads, 
                                                        dim_feedforward=self.hidden_dim, 
                                                        batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.fs_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 
                                             num_layers=self.n_layers, 
                                             norm=self.decoder_norm)
        

        # MLP prediction heads
        self.md_head = MLP_Head(input_dim=self.fs_dim,
                                hidden_dim=self.fs_dim,
                                output_dim=1)
        
        self.fa_head = MLP_Head(input_dim=self.fs_dim,
                                hidden_dim=self.fs_dim,
                                output_dim=1)
        
        self.di_head = MLP_Head(input_dim=self.fs_dim,
                                hidden_dim=self.fs_dim,
                                output_dim=3)
        
        self.w_head = MLP_Head(input_dim=self.fs_dim,
                                hidden_dim=self.fs_dim,
                                output_dim=1)

        if self.combined_mlp_head:
            # Existence score head with concatenated queries
            red_dim = self.fs_dim // 4
            self.qu_red = nn.Sequential(nn.Linear(self.fs_dim, red_dim),
                                        nn.LayerNorm(red_dim),
                                        nn.ReLU())
            
            self.extnc_head = MLP_Head(input_dim = red_dim * self.n_queries,
                                    hidden_dim = self.fs_dim,
                                    output_dim = self.n_queries)

        else:
            # Existence score head with separate query predictions
            self.extnc_head = MLP_Head(input_dim=self.fs_dim,
                                        hidden_dim=self.fs_dim,
                                        output_dim=1)


    def forward(self, X):
        B = X.size(0)

        # encoded signal
        enc_embed = self.encoder(X)


        # queries
        query_batch = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        enc_batch = enc_embed.unsqueeze(1)


        # decoder
        layer_outputs = []

        hs = query_batch
        for i in range(self.n_layers):
            hs = self.decoder.layers[i](tgt=hs, memory=enc_batch)

            # prediction on decoder layer for auxiliary loss
            if self.aux_loss:
                layer_outputs.append(self.make_predictions(hs))

        # computing predictions on final decoder block with layer normalization
        hs = self.decoder.norm(hs)
        final_output = self.make_predictions(hs)

        if self.aux_loss:
            return {
                'pred': final_output,
                'aux': layer_outputs
            }
        else:
            return final_output


    def make_predictions(self, hs):

        # normalized predictions for spectrum metrics 
        md = F.sigmoid(self.md_head(hs)) * 4.0
        fa = F.sigmoid(self.fa_head(hs))
        dir = F.normalize(self.di_head(hs), dim=-1)
        w = F.sigmoid(self.w_head(hs))

        if self.combined_mlp_head:
            # predictions for existence scores on concatenated queriy vectors
            hs_red = self.qu_red(hs)
            hs_cat = hs_red.reshape(hs.size(0), -1)

            extnc = self.extnc_head(hs_cat).unsqueeze(2)

        else:
            # predictions for existence scores on separate query vectors
            extnc = self.extnc_head(hs)


        return torch.cat([md, fa, dir, w, extnc], dim=-1)

