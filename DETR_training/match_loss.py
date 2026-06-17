import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torchvision
import numpy as np


class HungarianLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        # loss component weights
        self.md_lmd = args.md_loss_weight
        self.fa_lmd = args.fa_loss_weight
        self.dir_lmd = args.dir_loss_weight
        self.wt_lmd = args.wt_loss_weight
        self.exs_lmd = args.exs_loss_weight

        self.use_focal_loss = args.use_focal_loss
        print(f"Using focal loss: {self.use_focal_loss}")
        self.focal_alpha = args.focal_alpha
        self.focal_gamma = args.focal_gamma

        self.use_mape_weight_loss = args.use_mape_weight_loss
        print(f"Using MAPE weight loss: {self.use_mape_weight_loss}")

        self.use_normalized_angular_loss = args.use_normalized_angular_loss
        print(f"Using normalized angular loss: {self.use_normalized_angular_loss}")
        
        self.weighted_loss = args.weighted_loss
        print(f"Using weighted loss: {self.weighted_loss}")

        self.weighted_direction_loss = args.weighted_direction_loss
        print(f"Using weighted direction loss: {self.weighted_direction_loss}")

    def forward(self, y_pred, y_true, n_comps):
        """
        DETR-based loss function using Hungarian matching for computing the
        costs of MD, FA, direction vectors, compartment weights, and the 
        query existence score.

        :param y_pred:  Prediction tensor of shape (B, n_Q, 7). For each query,
                        the model predicts, in this exact order:
                        MD, FA, direction coordinates (x, y, z),
                        compartment weight, and existence score.

        :param y_true:  Target tensor of shape (B, n_compartments * 6). For each
                        compartment, the target values consist of MD, FA,
                        direction coordinates (x, y, z) and compartment weight in this exact order. All 
                        compartments of a sample are flattened into a single
                        vector. Samples with fewer compartments than the
                        dataset maximum may be padded; non-existent
                        compartments are filtered out using `n_comps`.

        :param n_comps: 1D tensor specifying the number of valid target
                        compartments per sample.
        """

        # loss initialization
        device = y_pred.device
        batch_size, columns = y_true.shape
        n_pred = y_pred.shape[-1] - 1 # last column is existence score, not part of target structure
        y_true = y_true.reshape(batch_size, int(columns/n_pred), n_pred)


        # cost computation of MD, FA, and weights withs mean squared error
        md_loss = ((y_pred[:,:,0].unsqueeze(2) - y_true[:,:,0].unsqueeze(1))/4.0)**2
        fa_loss = (y_pred[:,:,1].unsqueeze(2) - y_true[:,:,1].unsqueeze(1))**2

        if self.use_mape_weight_loss:
            wt_loss = torch.abs((y_pred[:,:,5].unsqueeze(2) - y_true[:,:,5].unsqueeze(1)) / (y_true[:,:,5].unsqueeze(1) + 1e-8))
        else:
            wt_loss = (y_pred[:,:,5].unsqueeze(2) - y_true[:,:,5].unsqueeze(1))**2


        # direction costs computed with cosine similarity and weighted with the target's FA
        if self.use_normalized_angular_loss:
            def normalized_symmetric_angular_loss(angle_deg_tensor):
                """
                Computes a normalized loss based on the angle in degrees using tensors.
                The loss is symmetric around 90 degrees and normalized to [0, 1].
                """
                normalized_angle = torch.where(angle_deg_tensor > 90.0, 180.0 - angle_deg_tensor, angle_deg_tensor)
                return normalized_angle / 90.0

            # apply the normalized symmetric angular loss to the angles derived from cosine similarity
            dir_sim = F.cosine_similarity(y_pred[:,:,2:5].unsqueeze(2), 
                                        y_true[:,:,2:5].unsqueeze(1), 
                                        dim=-1) 
            # clamp tightly to avoid acos +/- 1 which leads to NaN gradients during backprop
            dir_angle = torch.acos(torch.clamp(dir_sim, -1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / np.pi)  # Convert to degrees

            if self.weighted_direction_loss:
                dir_loss = normalized_symmetric_angular_loss(dir_angle) * y_true[:,:,1].unsqueeze(1)
            else: 
                dir_loss = normalized_symmetric_angular_loss(dir_angle)
        
        else:
            dir_sim = F.cosine_similarity(y_pred[:,:,2:5].unsqueeze(2), 
                                        y_true[:,:,2:5].unsqueeze(1), 
                                        dim=-1)

            if self.weighted_direction_loss:
                dir_loss = (1 - torch.abs(dir_sim)) * y_true[:,:,1].unsqueeze(1)
            else:
                dir_loss = (1 - torch.abs(dir_sim))


        # cost matrix for all prediction-target combinations, with MD, FA, and direction costs
        cost_matrix_m = (self.md_lmd * md_loss + 
                        self.fa_lmd * fa_loss + 
                        self.dir_lmd * dir_loss)
        
        if self.weighted_loss:
            # prediction-target combinations weighted on target weight
            cost_matrix_weighted = cost_matrix_m * y_true[:,:,5].unsqueeze(1)
        else:
            cost_matrix_weighted = cost_matrix_m

        # adding weight costs to weighted MD, Fa, direction costs
        cost_matrix = cost_matrix_weighted + self.wt_lmd * wt_loss


        # normalized existence score costs
        if self.use_focal_loss:
            extnc_scores = torch.sigmoid(y_pred[:,:,6])
            
            # Focal cost matching formula: -alpha * (1-p)**gamma * log(p) 
            # (With a small epsilon to avoid log(0))
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_class = (1 - alpha) * (extnc_scores ** gamma) * (-(1 - extnc_scores + 1e-8).log())
            pos_cost_class = alpha * ((1 - extnc_scores) ** gamma) * (-(extnc_scores + 1e-8).log())
            
            # The cost is the difference (cost of predicting 'exists' minus 'does not exist')
            extnc_cost = (pos_cost_class - neg_cost_class).unsqueeze(2).expand(-1, -1, y_true.size(1))
            
            # final cost matrix for matching
            cost_matrix = cost_matrix + self.exs_lmd * extnc_cost

        else:
            extnc_scores = torch.sigmoid(y_pred[:,:,6])
            extnc_cost = (1 - extnc_scores).unsqueeze(2).expand(-1, -1, y_true.size(1))

            # final cost matrix for matching
            cost_matrix = cost_matrix + self.exs_lmd * extnc_cost
        

        # cpu transfer
        cost_np = cost_matrix.detach().cpu().numpy()
        n_queries = y_pred.size(1)
        
        # Hungarian matching indeces
        all_pred_idx = []
        all_true_idx = []
        all_batch_idx = []
        
        for b in range(batch_size):

            # filtering paddings by the number of compartments 
            n_comp = n_comps[b].item() if torch.is_tensor(n_comps[b]) else n_comps[b]
            real_costs = cost_np[b, :, :n_comp]

            # matching optimal prediction-target assignments
            row_idx, col_idx = linear_sum_assignment(real_costs)
            
            all_pred_idx.extend(row_idx)
            all_true_idx.extend(col_idx)
            all_batch_idx.extend([b] * len(row_idx))

        
        # laoding indices on gpu if defined
        pred_indices = torch.tensor(all_pred_idx, device=device, dtype=torch.long)
        true_indices = torch.tensor(all_true_idx, device=device, dtype=torch.long)
        batch_indices = torch.tensor(all_batch_idx, device=device, dtype=torch.long)
        
        # metric's costs of the computed matches
        matched_md = md_loss[batch_indices, pred_indices, true_indices]
        matched_fa = fa_loss[batch_indices, pred_indices, true_indices]
        matched_dir = dir_loss[batch_indices, pred_indices, true_indices]
        matched_wt = wt_loss[batch_indices, pred_indices, true_indices]
        batch_weights = y_true[batch_indices, true_indices, 5]
        
        # weighting costs with target weights
        weighted_md = matched_md * batch_weights * self.md_lmd
        weighted_fa = matched_fa * batch_weights * self.fa_lmd
        weighted_dir = matched_dir * batch_weights * self.dir_lmd
        weighted_wt = matched_wt * self.wt_lmd
        

        # creating existence score labels, with 1 for matched predictions and 0 for the rest
        extnc = torch.zeros(batch_size, n_queries, device=device)
        extnc[batch_indices, pred_indices] = 1

        if self.use_focal_loss:
            # focal loss with configured alpha and gamma values
            cls_loss = torchvision.ops.sigmoid_focal_loss(
                inputs=y_pred[:,:,6], 
                targets=extnc, 
                alpha=self.focal_alpha, 
                gamma=self.focal_gamma, 
                reduction="mean"
            )
            
        else:
            # positives weighted up for class balancing
            p_weight = (n_queries - extnc.sum(dim=-1)) / extnc.sum(dim=-1)
            p_weight = torch.clamp(p_weight, min=0.5, max=10.0)
                
            # binary cross entropy as existence score loss
            weighted_bce = nn.BCEWithLogitsLoss(pos_weight=p_weight.unsqueeze(1))
            cls_loss = weighted_bce(y_pred[:,:,6], extnc)
        

        # computing average metric costs per sample
        batch_md_losses = torch.zeros(batch_size, device=device)
        batch_fa_losses = torch.zeros(batch_size, device=device)
        batch_dir_losses = torch.zeros(batch_size, device=device)
        batch_wt_losses = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            b_mask = batch_indices == b

            if b_mask.any():
                batch_md_losses[b] = weighted_md[b_mask].mean()
                batch_fa_losses[b] = weighted_fa[b_mask].mean()
                batch_dir_losses[b] = weighted_dir[b_mask].mean()
                batch_wt_losses[b] = weighted_wt[b_mask].mean()
        
        
        # final loss summes up metrics' costs of found matches and existence score loss per sample
        batch_losses = (batch_md_losses + batch_fa_losses + 
                       batch_dir_losses + batch_wt_losses + 
                       self.exs_lmd * cls_loss)
        

        # return avaerage losses of batch
        loss = batch_losses.mean()
        md_mean = batch_md_losses.mean()
        fa_mean = batch_fa_losses.mean()
        dir_mean = batch_dir_losses.mean()
        wt_mean = batch_wt_losses.mean()
        extnc_mean = self.exs_lmd * cls_loss
        
        return loss, md_mean, fa_mean, dir_mean, wt_mean, extnc_mean
    
