#!/usr/bin/env python
# coding: utf-8

# ### This notebook presents the architectures of the three recommendation systems tested within this framework
# # 1. Imports

# In[1]:


import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Commented out, and os import will be removed
# export_dir = os.getcwd() # Removed, appeared unused
import torch
import torch.nn as nn
import torch.nn.functional as F


# # 2. MLP recommender Architecture

# In[2]:


class MLP(nn.Module):
    def __init__(self, hidden_size, **kw):
        super(MLP, self).__init__()
        user_size = kw['num_items']
        item_size = kw['num_items']
        self.device = kw['device']
        self.users_fc = nn.Linear(user_size, hidden_size, bias = True).to(self.device)
        self.items_fc = nn.Linear(item_size, hidden_size, bias = True).to(self.device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_tensor, item_tensor):
        user_vec = self.users_fc(user_tensor.to(self.device))
        item_vec = self.items_fc(item_tensor.to(self.device))
        output = torch.matmul(user_vec, item_vec.T).to(self.device)
        return self.sigmoid(output).to(self.device)


# # 3. VAE recommender Architecture

# In[3]:


class VAE(nn.Module):
    def __init__(self, model_conf, **kw):
        super().__init__()
        self.device = kw['device']
        num_features = kw['num_features'] 
        num_items = kw['num_items'] 
        self.demographic = kw['demographic'] 
        if self.demographic:
            self.num_items = num_features
            self.items_only = num_items
        else:
            self.num_items = num_items
        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]
        self.dropout = model_conf['dropout']
        self.softmax = nn.Softmax(dim=1)
        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0
        
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.ReLU())
                
        self.to(self.device)

    def forward(self, rating_matrix, return_latent=False):
        """
        Forward pass with option to return latent variables
        
        Args:
            rating_matrix: Input rating matrix
            return_latent: If True, returns reconstruction, mean and logvar. If False, returns only reconstruction
        """
        # Encoder forward pass
        if len(rating_matrix.shape) == 1:
            rating_matrix = torch.unsqueeze(rating_matrix, 0)
        h = F.dropout(F.normalize(rating_matrix, dim=-1), p=self.dropout, training=self.training)
        
        for layer in self.encoder:
            h = layer(h)
    
        # Sample from latent space
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]
        std_q = torch.exp(0.5 * logvar_q)
        
        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=1.0)  # Changed std to 1.0
        sampled_z = mu_q + self.training * epsilon * std_q
        
        output = sampled_z
        for layer in self.decoder:
            output = layer(output)
            
        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return self.softmax(output), kl_loss, mu_q, std_q  # Return consistent outputs
        else:
            if self.demographic:
                return self.softmax(output[:,:self.items_only])
            return self.softmax(output)
        
    def train_one_epoch(self, dataset, optimizer, batch_size, alpha=0.5):
        """
        Train model for one epoch
        """
        self.train()
        train_matrix = dataset
        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))
        perm = np.random.permutation(num_training)
        loss = 0.0
    
        for b in range(num_batches):
            optimizer.zero_grad()
    
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx]).to(self.device)
    
            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap
    
            # Get reconstructions, mean, and logvar from forward pass
            pred_matrix, mu_q, logvar_q = self.forward(batch_matrix, return_latent=True)
    
            # Calculate losses
            # Cross entropy loss
            total_ce = -(F.log_softmax(pred_matrix, 1) * batch_matrix)
            ce_hist = total_ce[:,:self.num_items].sum(1).mean()
            ce_demo = total_ce[:,self.num_items:].sum(1).mean() if self.demographic else 0
            ce_loss = ce_hist + alpha * ce_demo
    
            # KL divergence loss
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
    
            # Total loss
            batch_loss = ce_loss + kl_loss * self.anneal
    
            batch_loss.backward()
            optimizer.step()
    
            self.update_count += 1
            loss += batch_loss
    
        return loss

    def predict(self, eval_users, test_batch_size):
        """
        Predict the model on test set
        :param eval_users: evaluation (test) user
        :param eval_pos: position of the evaluated (test) item
        :param test_batch_size: batch size for test set
        :return: predictions
        """
        with torch.no_grad():
            input_matrix = torch.Tensor(eval_users).to(self.device)
            preds = np.zeros_like(input_matrix.cpu())

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                    
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                batch_pred_matrix = batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        return preds
    


# Enhanced VAE Recommender with Dynamic Epoch Selection

# In[4]:


# import torch # REMOVE DUPLICATE
# import torch.nn as nn # REMOVE DUPLICATE
# import torch.nn.functional as F # REMOVE DUPLICATE
# import numpy as np # REMOVE DUPLICATE

class EnhancedVAE(nn.Module):
    def __init__(self, model_conf, **kw):
        super(EnhancedVAE, self).__init__()
        self.device = kw['device']
        num_features = kw['num_features']
        num_items = kw['num_items']
        self.demographic = kw['demographic']
        
        if self.demographic:
            self.num_items = num_features
            self.items_only = num_items
        else:
            self.num_items = num_items
            
        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]
        self.dropout = model_conf['dropout']
        self.softmax = nn.Softmax(dim=1)
        
        # Training configuration
        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']
        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0
        
        # Early stopping configuration
        self.patience = model_conf.get('patience', 5)
        self.min_delta = model_conf.get('min_delta', 0.001)
        self.performance_threshold = model_conf.get('performance_threshold', 0.20)
        
        # Initialize encoder
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.ReLU())

        # Initialize decoder
        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.ReLU())

        self.to(self.device)

    def forward(self, rating_matrix):
        # Encoder forward pass
        if len(rating_matrix.shape) == 1:
            rating_matrix = torch.unsqueeze(rating_matrix, 0)
        h = F.dropout(F.normalize(rating_matrix, dim=-1), p=self.dropout, training=self.training)
        
        for layer in self.encoder:
            h = layer(h)

        # Sample from latent space
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]
        std_q = torch.exp(0.5 * logvar_q)
        
        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        # Decoder forward pass
        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            if self.demographic:
                return self.softmax(output[:,:self.items_only])
            else:
                return self.softmax(output)

    def train_with_dynamic_epochs(self, train_data, valid_data, optimizer, batch_size, max_epochs=100, alpha=0.5):
        """
        Train the model with dynamic epoch selection based on performance criteria
        """
        best_metric = float('-inf')
        patience_counter = 0
        best_epoch = 0
        training_history = []
        
        for epoch in range(max_epochs):
            # Train for one epoch
            train_loss = self.train_one_epoch(train_data, optimizer, batch_size, alpha)
            
            # Evaluate on validation set
            hr10, hr50, hr100, mrr, mpr = self.evaluate(valid_data)
            current_metric = hr10  # Using HR@10 as primary metric
            
            # Store training history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'hr10': hr10,
                'hr50': hr50,
                'hr100': hr100,
                'mrr': mrr,
                'mpr': mpr
            })
            
            # Check if performance meets threshold criteria
            if current_metric > best_metric + self.min_delta:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
            
            # Early stopping checks
            if patience_counter >= self.patience:
                break
                
            # Performance threshold check
            if current_metric >= self.performance_threshold:
                break
                
        # Restore best model state
        self.load_state_dict(best_model_state)
        return best_epoch, training_history

    def evaluate(self, eval_data, batch_size=128):
        """
        Evaluate the model on validation/test data
        """
        self.eval()
        with torch.no_grad():
            # Implement evaluation metrics calculation
            hr10, hr50, hr100, mrr, mpr = 0.0, 0.0, 0.0, 0.0, 0.0
            num_users = len(eval_data)
            
            for i in range(0, num_users, batch_size):
                batch_users = eval_data[i:min(i + batch_size, num_users)]
                batch_predictions = self.forward(torch.Tensor(batch_users).to(self.device))
                
                # Calculate metrics for batch
                batch_hr10 = self.calculate_hit_ratio(batch_predictions, k=10)
                batch_hr50 = self.calculate_hit_ratio(batch_predictions, k=50)
                batch_hr100 = self.calculate_hit_ratio(batch_predictions, k=100)
                batch_mrr = self.calculate_mrr(batch_predictions)
                batch_mpr = self.calculate_mpr(batch_predictions)
                
                # Accumulate metrics
                hr10 += batch_hr10
                hr50 += batch_hr50
                hr100 += batch_hr100
                mrr += batch_mrr
                mpr += batch_mpr
            
            # Average metrics
            hr10 /= num_users
            hr50 /= num_users
            hr100 /= num_users
            mrr /= num_users
            mpr /= num_users
            
        return hr10, hr50, hr100, mrr, mpr

    def calculate_hit_ratio(self, predictions, k):
        """Calculate Hit Ratio @ k"""
        _, top_k = torch.topk(predictions, k, dim=1)
        return float(torch.any(top_k == self.target_items.unsqueeze(1), dim=1).float().mean())

    def calculate_mrr(self, predictions):
        """Calculate Mean Reciprocal Rank"""
        ranks = torch.argmax(predictions == self.target_items.unsqueeze(1), dim=1).float() + 1
        return float((1.0 / ranks).mean())

    def calculate_mpr(self, predictions):
        """Calculate Mean Percentile Rank"""
        ranks = torch.argmax(predictions == self.target_items.unsqueeze(1), dim=1).float() + 1
        return float((ranks / predictions.size(1)).mean())


# # 4. NCF

# In[5]:


class GMF_model(nn.Module):
    def __init__(self, hidden_size=8, **kw):
        super(GMF_model, self).__init__()
        self.device = kw['device']
        user_size = kw['num_features']
        item_size = kw['num_items']
        self.embed_user_GMF = nn.Linear(user_size, hidden_size, bias = False).to(self.device)
        self.embed_item_GMF = nn.Linear(item_size, hidden_size, bias = False).to(self.device)
        self.predict_layer = nn.Linear(hidden_size, 1, bias = True).to(self.device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_tensor, item_tensor):
        user_vec = self.embed_user_GMF(user_tensor.to(self.device))
        item_vec = self.embed_item_GMF(item_tensor.to(self.device))
        if user_vec.shape!=item_vec.shape:
            user_res = torch.zeros(item_vec.shape).to(self.device)
            user_res[:] = user_vec
            user_vec = user_res
            
        output = self.predict_layer(torch.mul(user_vec, item_vec))
        
        return self.sigmoid(output)


# In[6]:


class MLP_model(nn.Module):
    def __init__(self, hidden_size, num_layers, **kw):
        super(MLP_model, self).__init__()
        self.device = kw['device']
        user_size = kw['num_features']
        item_size = kw['num_items']
        factor_num = hidden_size
        self.embed_user_MLP = nn.Linear(user_size, factor_num * (2 ** (num_layers - 1)), bias = False).to(self.device)
        self.embed_item_MLP = nn.Linear(item_size, factor_num * (2 ** (num_layers - 1)), bias = False).to(self.device)
        
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=0.5))
            MLP_modules.append(nn.Linear(input_size, input_size//2).to(self.device))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        self.predict_layer = nn.Linear(hidden_size, 1, bias = True).to(self.device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_tensor, item_tensor):
        embed_user_MLP = self.embed_user_MLP(user_tensor.to(self.device))
        embed_item_MLP = self.embed_item_MLP(item_tensor.to(self.device))
        if embed_user_MLP.shape!=embed_item_MLP.shape:
            user_res = torch.zeros(embed_item_MLP.shape).to(self.device)
            user_res[:] = embed_user_MLP
            embed_user_MLP = user_res
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        output = self.predict_layer(output_MLP)
        return self.sigmoid(output)


# In[7]:


class NCF(nn.Module):
    def __init__(self, factor_num, num_layers,
                    dropout, model, GMF_model=None, MLP_model=None, **kw):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """        
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.device = kw['device']
        user_size = kw['num_features']
        item_size = kw['num_items']
        self.embed_user_GMF = nn.Linear(user_size, factor_num, bias = False)
        self.embed_item_GMF = nn.Linear(item_size, factor_num, bias = False)
        self.embed_user_MLP = nn.Linear(
                user_size, factor_num * (2 ** (num_layers - 1)), bias = False)
        self.embed_item_MLP = nn.Linear(
                item_size, factor_num * (2 ** (num_layers - 1)), bias = False)

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight_()
        
        # Move the entire model to the specified device
        self.to(self.device)

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                            self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                            self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                            self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                            self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                        self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        user = user.to(self.device)
        item = item.to(self.device)
        
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            if embed_user_GMF.shape!=embed_item_GMF.shape:
                user_res = torch.zeros(embed_item_GMF.shape, device=self.device)
                user_res[:] = embed_user_GMF
                embed_user_GMF = user_res
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            if embed_user_MLP.shape!=embed_item_MLP.shape:
                user_res = torch.zeros(embed_item_MLP.shape, device=self.device)
                user_res[:] = embed_user_MLP
                embed_user_MLP = user_res
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        prediction = self.sigmoid(prediction)
        return prediction.view(-1)


# # LXR Explainer Model Architecture
class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size, device_passed):
        super(Explainer, self).__init__()
        self.device = device_passed # Use passed device
        
        self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(self.device)
        self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(self.device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(self.device),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=user_size).to(self.device),
            nn.Sigmoid()
        ).to(self.device)
        
    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(self.device)
        return expl_scores

