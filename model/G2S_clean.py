from typing import Mapping
import numpy as np
import networkx as nx
#import igraph
import torch
#from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical
from torch_geometric.nn import MessagePassing, global_mean_pool

import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear
import torch.nn.functional as F
#from torch.nn import Parameter
#from torch.nn.init import xavier_uniform
from torch import embedding_renorm_, scatter 

from torch_geometric.data import Data
#from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.utils import to_dense_adj

#from onmt.decoders import TransformerDecoder
from model.transformer_mod import TransformerDecoder, TransformerLMDecoder
#from onmt.modules.embeddings import Embeddings
from model.embeddings_mod import Embeddings
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from data_processing.data_utils import *


class Lin_layer_MP(torch.nn.Module):
    '''
    Linear NN used in the weighted edge centered MP from scratch
    '''

    def __init__(self, in_channels=300, out_channels=300):
        super(Lin_layer_MP, self).__init__()

        # Define linear layer and relu
        self.lin = Linear(in_channels, out_channels)
        self.relu = ReLU()

    def forward(self, h0, weighted_sum):
        x = self.lin(weighted_sum)
        h1 = self.relu(h0 + x)

        return h1


class vec_scratch_MP_layer(torch.nn.Module):
    '''
    This is a vectorized implementation of the edge centered message passing
    '''

    def __init__(self, in_channels=300, out_channels=300):
        super(vec_scratch_MP_layer, self).__init__()

        # Define linear layer and relu
        self.lin = Linear(in_channels, out_channels)
        self.relu = ReLU()

    def forward(self, h0, dest_is_origin_matrix, dev):
        # pass weighted sum through a NN to obtain new featurization of that edge
        weighted_sum = torch.sparse.mm(
            dest_is_origin_matrix.to(dev), h0.to(dev))
        x = self.lin(weighted_sum)
        h_next = self.relu(h0+x)

        return h_next


class Lin_layer_node(torch.nn.Module):
    '''
    Linear NN used in the weighted atom updater
    '''

    def __init__(self, in_channels, out_channels=300):
        super(Lin_layer_node, self).__init__()

        # Define linear layer and relu
        self.lin = Sequential(Linear(in_channels, out_channels),
                              ReLU())

    def forward(self, concat_atom_edges):
        atom_hidden = self.lin(concat_atom_edges)

        return atom_hidden


class vec_atom_updater(torch.nn.Module):
    '''
    This is a vectorized version of the atom update step
    '''

    def __init__(self, in_channels, out_channels=300, output_layer=False):
        super(vec_atom_updater, self).__init__()

        # Define linear layer and relu
        if not output_layer:
            self.lin = Sequential(Linear(in_channels, out_channels),
                              ReLU())
        else: 
            self.lin = Linear(in_channels, out_channels)

    def forward(self, nodes, h, inc_edges_to_atom_matrix, device):
        sum_inc_edges = torch.sparse.mm(inc_edges_to_atom_matrix.to(device), h)
        atom_embeddings = torch.cat((nodes.to(device), sum_inc_edges), dim=1)
        # pass through NN
        atom_updates = self.lin(atom_embeddings)
        return atom_updates

# %% Define models


class Wdmpnn_Conv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, device, first_linear=True):
        super(Wdmpnn_Conv, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.first_linear = first_linear    

        self.lin1 = Sequential(Linear(node_dim + edge_dim, hidden_dim),
                                ReLU(),
                                ).to(device)
        
        self.lin2 = Sequential(Linear(hidden_dim + edge_dim, hidden_dim),
                                ReLU(),
                                ).to(device)
        

        # define edge message passing layers
        self.vec_scratch_MP_layer1 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)
        self.vec_scratch_MP_layer2 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)
        self.vec_scratch_MP_layer3 = vec_scratch_MP_layer(
            in_channels=hidden_dim, out_channels=hidden_dim).to(device)

        # define node message passing layer
        self.vec_atom_updater = vec_atom_updater(
            in_channels=node_dim+hidden_dim, out_channels=hidden_dim).to(device)
        
        # define node message passing layer2
        self.vec_atom_updater2 = vec_atom_updater(
            in_channels=hidden_dim+hidden_dim, out_channels=hidden_dim, output_layer=True).to(device)

    def forward(self, graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        
        # only in the first network with shared layers
        if self.first_linear:
            nodes = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            # Repeat the node features for each edge
            nodes_to_edge = nodes[edge_index[0]]
            # Initialize the edge features with the concatenation of the node and edge features
            h0 = torch.cat([nodes_to_edge, edge_attr], dim=1)
        
            # Pass this through a NN to compute the initialize hidden features
            h0 = self.lin1(h0)

            # pass the messages along edges
            h1 = self.vec_scratch_MP_layer1(h0, dest_is_origin_matrix, device)
            h2 = self.vec_scratch_MP_layer2(h1, dest_is_origin_matrix, device)
            h3 = self.vec_scratch_MP_layer3(h2, dest_is_origin_matrix, device)

            # get atom embeddings by summing over all incoming edges and concatenating with original atom features
            atom_embeddings = self.vec_atom_updater(
                nodes, h3, inc_edges_to_atom_matrix, device)
            # atom embeddings 
            
            return atom_embeddings
        
        else:
            nodes = graph.shared_output
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            # Repeat the node features for each edge
            nodes_to_edge = nodes[edge_index[0]]
            # Initialize the edge features with the concatenation of the node and edge features
            h0 = torch.cat([nodes_to_edge, edge_attr], dim=1)

            h0 = self.lin2(h0)
            # pass the messages along edges
            h1 = self.vec_scratch_MP_layer1(h0, dest_is_origin_matrix, device)
            h2 = self.vec_scratch_MP_layer2(h1, dest_is_origin_matrix, device)
            h3 = self.vec_scratch_MP_layer3(h2, dest_is_origin_matrix, device)

            # get atom embeddings by summing over all incoming edges and concatenating with original atom features
            atom_embeddings = self.vec_atom_updater2(
                nodes, h3, inc_edges_to_atom_matrix, device)
            
            return atom_embeddings

class GraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, device, model_config):
        super(GraphEncoder, self).__init__()

        self.tconv1 = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device)
        
        self.mu = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device, first_linear=False)
        self.logvar = Wdmpnn_Conv(node_dim, edge_dim, hidden_dim, device, first_linear=False)
    
    def forward(self, graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):

        atom_weights = graph.W_atoms

        shared_output = self.tconv1(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

        graph.shared_output = shared_output

        mu = self.mu(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        logvar = self.logvar(graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

        mu = global_mean_pool(mu * atom_weights.view(-1, 1), graph.batch).to(device)
        logvar = global_mean_pool(logvar * atom_weights.view(-1, 1), graph.batch).to(device)
        #print(torch.mean(mu), torch.mean(logvar))

        return mu, logvar


## Transformer decoder
class SequenceDecoder(nn.Module):
    def __init__(self, model_config, vocab, loss_weights, add_latent):
        """Implementation of transformer decoder

        Args:
            model_config (Dict): model config settings
            data_config (Dict): data config settings
            vocab (Dict): complete vocab of tokens
        """
        super().__init__()
        
        self.ndim= model_config['embedding_dim']
        self.config = model_config
        self.max_n=256 #TODO: this should not be hardcoded
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.beam_size = 1
        self.add_latent = add_latent

        self.decoder_embeddings = Embeddings(
            word_vec_size=model_config['embedding_dim'],
            word_vocab_size=len(self.vocab)+1,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=0.3
        )
        if self.add_latent: 
            d_model=model_config['embedding_dim']*2
        else: 
            d_model=model_config['embedding_dim']
        #TransformerDecoder (with EDATT) or TransformerLMDecoder (without EDAtt)
        self.Decoder = TransformerDecoder(num_layers=model_config['decoder_num_layers'], \
            d_model=d_model, heads=model_config['num_attention_heads'], \
            d_ff=2048, copy_attn=False, self_attn_type="scaled-dot", dropout=0.3, attention_dropout=0.3, \
            embeddings=self.decoder_embeddings, max_relative_positions=4, aan_useffn=False, \
            full_context_alignment=False, alignment_layer=-3, alignment_heads=0
        )

        self.output_layer = nn.Linear(d_model, len(self.vocab), bias=True)

        if self.config['loss']=="ce" or self.config['loss']=="wce":
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.vocab["_PAD"],
                reduction="mean",
                weight=loss_weights
            )
        elif self.config['loss']=="focal":
            self.criterion = FocalLoss(
                gamma = 1,
                ignore_index=self.vocab["_PAD"],
                reduction="none"
            )

    def forward(self, graph_batch, z, loss_weights=None):
        """Forward pass of decoder

        Args:
            z (Tensor): Latent space embedding [b, h]
            graph_batch (Data): Data of correct graphs

        Returns:
            Tensor, Tensor: Reconstruction loss and accuracy
        """
        z_length = z.size(1)
        src_lengths = torch.ones(z.size(0), device=z.device).long()*z_length # long tensor [b,]
        # prepare target
        target = torch.tensor(graph_batch.tgt_token_ids, device=z.device)[:, :-1] #pop last token
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"]) #pads SOS token left side (beginning of sequences)
        target = m(target)
        target = target.unsqueeze(-1) 

        #TODO Check if this actually works, what happens if target size (max length tokens) is smaller than z hidden dim? --> no padding?
        # Why the padding? doesn't really need it?
        #m = torch.nn.ZeroPad2d((0, target.size(1) - z.size(1), 0, 0))
        #z_padded = m(z)
        #padded_memory_bank = z_padded.unsqueeze(1).transpose(1,2)
        # Alternativ
        enc_output = z.unsqueeze(1)
        self.Decoder.state["src"] = enc_output
    
        # decode
        dec_outs, _ = self.Decoder(tgt=target, enc_out=enc_output, src_len=src_lengths, step=target.size(1), add_latent=self.add_latent)
        #dec_outs, _ = self.Decoder(tgt=target, memory_bank=z.unsqueeze(2).repeat([1,1,56]).transpose(1,0), memory_lengths=memory_lens)
        dec_outs = self.output_layer(dec_outs)                                  # [t, b, h] => [t, b, v]
        dec_outs = dec_outs.permute(0, 2, 1)                                    # [t, b, v] => [b, v, t]

        # evaluate
        # reproducible on GPU requires reduction none in CE loss in order to use torch.use_deterministic_algorithms(True), mean afterwards
        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8
        target = torch.tensor(graph_batch.tgt_token_ids, device=z.device)
        recon_loss = self.criterion(
            input=dec_outs, 
            target=target.long()
        )
        # custom: assign more weight to predictions of critical decisions in string (stoichiometry and connectivity)
        #positional_weights = self.conditional_position_weights(torch.tensor(graph_batch.tgt_token_ids, device=z.device))
        #positional_weights = torch.from_numpy(positional_weights).to(z.device)

        #recon_loss = recon_loss * positional_weights
        #recon_loss = recon_loss.mean()
        
        predictions = torch.argmax(dec_outs.transpose(1,0), dim=0)                             # [b, t]
        mask = (target != self.vocab["_PAD"]).long()
        accs = (predictions == target).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()


        return recon_loss, acc, predictions, target
    

    def inference(self, z):
        global_scorer = GNMTGlobalScorer(alpha=0.5,
                                             beta=0.0,
                                             length_penalty="avg",
                                             coverage_penalty="none")

        """ decode_strategy = GreedySearch(
            pad=self.vocab["_PAD"],
            bos=self.vocab["_SOS"],
            eos=self.vocab["_EOS"],
            unk=self.vocab['_UNK'],
            ban_unk_token = True, # TODO: Check if true 
            global_scorer=global_scorer,
            keep_topp=1,
            beam_size=5,
            start=self.vocab["_SOS"], # can be either bos or eos token
            batch_size=z.size(0),
            min_length=1,
            max_length=self.max_n,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            return_attention=False,
            sampling_temp=0.0,
            keep_topk=1
        ) """
        
        decode_strategy = BeamSearch(
            pad=self.vocab["_PAD"],
            bos=self.vocab["_SOS"],
            eos=self.vocab["_EOS"],
            unk=self.vocab['_UNK'],
            ban_unk_token = True, # TODO: Check if true 
            global_scorer=global_scorer,
            beam_size=5,
            start=self.vocab["_SOS"], # can be either bos or eos token
            batch_size=z.size(0),
            min_length=1,
            n_best=1,
            stepwise_penalty=None,
            ratio=0.0,
            max_length=self.max_n,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            return_attention=False,
        )

        # adapted from onmt.translate.translator
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        
        # (1) Encoder output.
        z_length = z.size(1)
        src_lengths = torch.ones(z.size(0), device=z.device).long()*z_length # long tensor [b,]
        enc_output = z.unsqueeze(1)
        self.Decoder.state["src"] = enc_output
        
        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None

        fn_map_state, enc_out, src_len_tiled, src_map = decode_strategy.initialize(
            enc_out=enc_output,
            src_len=src_lengths,
            src_map=src_map,
            target_prefix=target_prefix,
            device=z.device

        )

        if fn_map_state is not None:
            self.Decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            #decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            decoder_input = decode_strategy.current_predictions.view(-1, 1, 1)
            dec_outs, dec_attn = self.Decoder(tgt=decoder_input, enc_out=enc_out, src_len=src_len_tiled, step=step, add_latent=self.add_latent)
            if "std" in dec_attn:
                attn = dec_attn["std"].detach()
            else:
                attn = None

            scores = self.output_layer(dec_outs.squeeze(1))
            log_probs = F.log_softmax(scores.to(torch.float32), dim=-1).detach()

            decode_strategy.advance(log_probs, attn)
            
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x.index_select(0, select_indices) for x in enc_out)
                else:
                    enc_out = enc_out.index_select(0, select_indices)

                src_len_tiled = src_len_tiled.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(0, select_indices)

            if parallel_paths > 1 or any_finished:
                self.Decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )
        
        #self.reset_layer_cache() # reset layers

        return decode_strategy.predictions
        
    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.Decoder.state["cache"] is not None:
            _recursive_map(self.Decoder.state["cache"])    

    def reset_layer_cache(self):
        """After inference, layer cache needs to be reset"""
        for layer in self.Decoder.transformer_layers:
            layer.self_attn.layer_cache = (False, {'keys': torch.tensor([]),
                                    'values': torch.tensor([])})
    def conditional_position_weights(self, batch):
        batch=batch.cpu()
        weights = np.ones_like(batch)    
        for nr, sample in enumerate(batch):

            # Define the preceding sequences and their corresponding weight multipliers
            preceding_sequences = [
                ["|", "0_0", "."], # stoichiometry decision
                ["|", "<", "1", "-"] ,# first decision connectivity
                ["|", "<", "1", "-", "3",":","0_0","."] ,# second decision connectivity
                
            ]
            # preceding sequences token ids
            preceding_sequences = [list(get_seq_features_from_line(preceding_sequence, vocab=self.vocab, max_tgt_len=len(preceding_sequence)+1)[0][:-1]) for preceding_sequence in preceding_sequences ]

            weight_multipliers = [2, 2]  # Weight multipliers for each preceding sequence
            # if preceding sequence, double the weight
            for i in range(len(sample)):
                for sequence, multiplier in zip(preceding_sequences, weight_multipliers):
                    if i >= len(sequence) and list(sample[i - len(sequence):i]) == sequence:
                        weights[nr,i] *= multiplier

        return weights

class G2S_VAE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']
        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)

    def sample(self, mean, log_var, eps_scale=1):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean  
              
    def sample_inference(self, mean, log_var, eps_scale=1):
        
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mean)   

    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))

        # decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z)

        return recon_loss + self.beta*kl_loss, recon_loss, kl_loss, acc, predictions, target, z
    

    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        #TODO: Function arguments (test batch?, single graph?, latent representation?), right encoder call
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample_inference(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        predictions = self.Decoder.inference(z)
               
        return predictions, mean, log_var, z
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    
class G2S_VAE_PPguided(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']

        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)
        
        self.pp_ffn_hidden = 56
        self.alpha = model_config['max_alpha']
        #self.alpha=0.1
        #self.max_n=data_config['max_num_nodes']
        self.PP_lin1 = Sequential(Linear(embedding_dim, self.pp_ffn_hidden), ReLU(), ).to(device)
        self.PP_lin2 = Sequential(Linear(self.pp_ffn_hidden, 2)).to(device)
        self.dropout = nn.Dropout(0.2)

    def sample(self, mean, log_var, eps_scale=0.01):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean

    def sample_inference(self, mean, log_var, eps_scale=1):
        
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mean)   


    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))

        # Property predictions 
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        pp_hidden = self.dropout(pp_hidden)
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, 2] for 2 properties
        y1 = torch.unsqueeze(batch_list.y1.float(),1)
        y2 = torch.unsqueeze(batch_list.y2.float(),1)
        y_true = torch.cat((y1,y2), dim=1)
        mse = self.masked_mse(y_true,y)

        # decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z)

        return recon_loss + self.beta*kl_loss + self.alpha*mse, recon_loss, kl_loss, mse, acc, predictions, target, z, y

    def masked_mse(self, y_true, y_pred):
        # Create a mask where the true values are not NaN
        mask = ~torch.isnan(y_true).any(dim=1)
        
        # Calculate MSE only for non-missing values
        mse = F.mse_loss(y_pred[mask], y_true[mask], reduction='none')
        
        # Take the mean over the non-missing values
        return torch.mean(mse)
    
    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        #TODO: Function arguments (test batch?, single graph?, latent representation?), right encoder call
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample_inference(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, 2] for 2 properties

        predictions = self.Decoder.inference(z)
        # Property predictions 
               
        return predictions, mean, log_var, z, y 
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    
class G2S_VAE_PPguideddisabled(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, embedding_dim, device, model_config, vocab, seed, loss_weights=None, add_latent=True):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.seed=seed
        self.eps = model_config['epsilon']

        try: 
            self.embedding_dim = model_config['embedding_dim']
        except:
            self.embedding_dim = embedding_dim
        # in case beta is schedule the value will be specified in train.py
        if not model_config["beta"] =="schedule":
            self.beta=1.0
        self.config = model_config
        self.vocab = vocab
        #self.max_n=data_config['max_num_nodes']
        #if model_config['pooling']=='custom':
        #    self.Encoder = GraphEncoder_GMT(node_dim, edge_dim, hidden_dim, device, model_config)
        #elif model_config['pooling']=='mean':
        self.Encoder = GraphEncoder(node_dim, edge_dim, hidden_dim, device, model_config)
        self.Decoder = SequenceDecoder(model_config, vocab, loss_weights, add_latent=add_latent)
        if not self.hidden_dim==self.embedding_dim:
            self.lincompress = Linear(self.hidden_dim, self.embedding_dim).to(device)
        
        self.pp_ffn_hidden = 56
        self.alpha = model_config['max_alpha']

        #self.max_n=data_config['max_num_nodes']
        self.PP_lin1 = Sequential(Linear(embedding_dim, self.pp_ffn_hidden), ReLU(), ).to(device)
        self.PP_lin2 = Sequential(Linear(self.pp_ffn_hidden, 2)).to(device)
        self.dropout = nn.Dropout(0.2)

    def sample(self, mean, log_var, eps_scale=0.01):
        
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mean)
        else:
            return mean        

    def forward(self, batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
        # encode
        h_G_mean, h_G_var = self.Encoder(batch_list, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not self.hidden_dim==self.embedding_dim:
            h_G_mean = self.lincompress(h_G_mean)
            h_G_var = self.lincompress(h_G_var)
        z = self.sample(h_G_mean, h_G_var, eps_scale=self.eps)
        kl_loss = -0.5 * torch.sum(1 + h_G_var - h_G_mean.pow(2) - h_G_var.exp())/(len(batch_list.ptr-1))

        # Property predictions 
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        pp_hidden = self.dropout(pp_hidden)
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, 2] for 2 properties
        y1 = torch.unsqueeze(batch_list.y1.float(),1)
        y2 = torch.unsqueeze(batch_list.y2.float(),1)
        y_true = torch.cat((y1,y2), dim=1)
        mse = self.masked_mse(y_true,y)

        # decode
        recon_loss, acc, predictions, target = self.Decoder(batch_list, z)
        
        # Notice that the MSE explicitely is not used in the aggregated overall loss, so the loss does not contribute to changing the parameters of encoder and decoder. 
        return recon_loss + self.beta*kl_loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y

    def masked_mse(self, y_true, y_pred):
        # Create a mask where the true values are not NaN
        mask = ~torch.isnan(y_true).any(dim=1)
        
        # Calculate MSE only for non-missing values
        mse = F.mse_loss(y_pred[mask], y_true[mask], reduction='none')
        
        # Take the mean over the non-missing values
        return torch.mean(mse)
    
    def inference(self, data, device, dest_is_origin_matrix=None, inc_edges_to_atom_matrix=None, sample=False, log_var=None):
        #TODO: Function arguments (test batch?, single graph?, latent representation?), right encoder call
        if isinstance(data, torch.Tensor): # tensor with latent representations
            if data.size(-1) != self.embedding_dim: #tensor input needs to be embedding/hidden size
                raise Exception('Size of input is {}, must be {}'.format(data.size(0), self.embedding_dim))
            if data.dim() == 1: # is the case if data is only one sample
                mean = data.unsqueeze(0) #dimension for batch size
            else:
                mean = data
        elif isinstance(data, Data): # batch list of graphs
            mean, log_var = self.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            if not self.hidden_dim==self.embedding_dim:
                mean = self.lincompress(mean)
                log_var = self.lincompress(log_var)
           
        if sample:
            z= self.sample(mean, log_var, eps_scale=self.eps)
        else:
            z= mean
            log_var = 0
       
        pp_hidden = self.PP_lin1(z) #[b,hidden_dim] -> [b,pp_ffn_hidden]
        y = self.PP_lin2(pp_hidden) #[b,pp_ffn_hidden] -> [b, 2] for 2 properties

        predictions = self.Decoder.inference(z)
        # Property predictions 
               
        return predictions, mean, log_var, z, y 
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets
     from https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b '''

    def __init__(self, gamma, scale_loss=10, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
        self.scale_loss = scale_loss

    def forward(self, input, target):
        cross_entropy = super().forward(input, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = loss*self.scale_loss
        if self.reduction == 'mean': 
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss