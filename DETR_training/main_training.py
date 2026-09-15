import torch

import argparse
import json
import numpy as np
import time
import datetime

import DETR_training.dl_models as models
import DETR_training.match_loss as Loss
from DETR_training.train_utils import Train_Utils



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)


    # paths
    parser.add_argument('--train_data_path', default="./datagen/data/train/comp_setup_dynamic__max_n_comp_5__min_n_comp_2__noise_0.01__n_signal_values_166.csv", type=str)
    parser.add_argument('--val_data_path', default="./datagen/data/validation/comp_setup_dynamic__max_n_comp_5__min_n_comp_2__noise_0.01__n_signal_values_166.csv", type=str)

    parser.add_argument('--save_path', default="./DETR_training/experiments/paper_model/", type=str)

    parser.add_argument('--model_path', default=None, type=str, help="path of trained model.")


    # train parameters
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_step', default=100000, type=int, help="step size for lr scheduler.")
    parser.add_argument('--w_decay', default=1e-4, type=float)
    parser.add_argument('--opt_betas', default=(0.9,0.98), type=tuple, help="AdamW parameters.")

    parser.add_argument('--b_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--early_stop_patience', default=100, type=int, help="early stopping patience.")
    
    parser.add_argument('--device', default="cuda", type=str)


    # model parameters
    parser.add_argument('--input_dim', default=166, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)

    parser.add_argument('--fs_dim', default=256, type=int, help="size of feature and query vectors.")
    parser.add_argument('--n_queries', default=40, type=int)

    parser.add_argument('--n_dlayers', default=4, type=int, help="number of decoder blocks.")
    parser.add_argument('--n_multihead', default=4, type=int, help="number of attention heads.")

    parser.add_argument('--encoder_type', default='Transformer', type=str, help="type of encoder to use: MLP, Transformer or TiRex.")


    # loss parameters
    parser.add_argument('--md_loss_weight', default=7.5, type=float, help="lamda weight of MD-loss.")
    parser.add_argument('--fa_loss_weight', default=20.0, type=float, help="lamda weight of FA-loss.")
    parser.add_argument('--dir_loss_weight', default=10.0, type=float, help="lamda weight of direction-loss.")
    parser.add_argument('--wt_loss_weight', default=0.4, type=float, help="lamda weight of weight-loss.")
    parser.add_argument('--exs_loss_weight', default=0.7, type=float, help="lamda weight of esxistence score-loss.")
    
    parser.add_argument('--use_focal_loss', default=True, type=bool, help="activating focal loss instead of BCE.")
    parser.add_argument('--focal_alpha', default=0.25, type=float, help="alpha parameter for focal loss.")
    parser.add_argument('--focal_gamma', default=2.0, type=float, help="gamma parameter for focal loss.")

    parser.add_argument('--aux_loss', default=False, type=bool, help="activating the auxiliary loss.")
    parser.add_argument('--aux_m', default=1.0, type=float, help="linear auxiliary loss weighting increasing for later decoder blocks.")

    parser.add_argument('--combined_mlp_head', default=True, type=bool, help="activating the combined MLP head for existence score prediction.")

    parser.add_argument('--use_mape_weight_loss', default=True, type=bool, help="activating MAPE loss for weight prediction instead of MSE.")

    parser.add_argument('--use_normalized_angular_loss', default=True, type=bool, help="activating normalized angular loss for direction prediction.")

    parser.add_argument('--weighted_loss', default=True, type=bool, help="activating weighting of MD, FA and direction loss on target weight.")

    parser.add_argument('--weighted_direction_loss', default=True, type=bool, help="activating weighting of direction loss on target FA value.")

    parser.add_argument('--seed', default=42, type=int, help="global seed for reproducibility.")

    return parser



def main(args):
    # set global seed
    import random
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start = time.time()

    # Construct save paths
    import os
    # extract dataset name from train_data_path and create subfolder within save_path
    dataset_name = os.path.basename(args.train_data_path).replace(".csv", "")
    args.save_path = os.path.join(args.save_path, dataset_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.model_save_path = os.path.join(args.save_path, "detr_model")
    args.log_save_path = os.path.join(args.save_path, "detr_logs")
    args.config_save_path = os.path.join(args.save_path, "detr_config.json")

    # model initialization
    model = models.DWI_DETR(args)

    # load weights of trained model if defined
    if args.model_path:
        models.load_state_dict_compat(model, torch.load(args.model_path, weights_only=True))

    model.to(args.device)


    # train initialization
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr= args.lr, 
                                  weight_decay= args.w_decay, 
                                  betas= args.opt_betas)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)
    
    criterion = Loss.HungarianLoss(args)

    setup = Train_Utils(model=model, 
                        optimizer=optimizer, 
                        criterion=criterion,
                        args=args)
    
    print("Model and training setup initialized.")

    # loading dataset in dataloader
    train_data = setup.get_data(args.train_data_path, args.b_size, train=True)
    val_data = setup.get_data(args.val_data_path, args.b_size, train=False)
    

    # loss dictionary with all components
    losses = {}
    losses["train_loss"] = []
    losses["val_loss"] = []
    losses["val_loss_md"] = []
    losses["val_loss_fa"] = []
    losses["val_loss_di"] = []
    losses["val_loss_wt"] = []
    losses["val_loss_extnc"] = []


    # check for auxiliary loss
    print("Auxiliary loss activated: ", args.aux_loss)

    print("Start training")

    # save config
    with open(args.config_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    best_val_loss = float('inf')
    patience_counter = 0

    # start training
    for epoch in range(args.epochs):

        # training and evaluation run for one epoch
        ep_loss, val_losses = setup.train_epoch(train_data, val_data, epoch)

        scheduler.step()


        # save aggregated results
        losses["train_loss"].append(ep_loss)
        losses["val_loss"].append(val_losses[0])
        losses["val_loss_md"].append(val_losses[1])
        losses["val_loss_fa"].append(val_losses[2])
        losses["val_loss_di"].append(val_losses[3])
        losses["val_loss_wt"].append(val_losses[4])
        losses["val_loss_extnc"].append(val_losses[5])


        # early stopping check
        val_loss = val_losses[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_save_path + "_best")
        else:
            patience_counter += 1

        # save model and log file every 50 epochs
        if epoch % 50 == 0 and epoch > 1:

            torch.save(model.state_dict(), args.model_save_path + f"_{epoch}ep")
            np.save(args.log_save_path + f"_{epoch}ep", losses)


        # epoch informations
        print(f"-{str(datetime.datetime.now())} " +
                f"- Epoche: {epoch+1:02d}/{args.epochs} " +
                f"- Train Loss: {ep_loss:.10f} " +
                f"- Val Loss: {val_losses[0]:.10f} " +
                f"- MD Loss: {val_losses[1]:.10f} " +
                f"- FA Loss: {val_losses[2]:.10f} " +
                f"- Direction Loss: {val_losses[3]:.10f} " +
                f"- Weight Loss: {val_losses[4]:.10f} " +
                f"- Existence Loss: {val_losses[5]:.10f}")
        

        # clearing gpu cache
        torch.cuda.empty_cache() 


        # apply early stopping
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


    # safe final model and losses
    torch.save(model.state_dict(), args.model_save_path+"_last")

    np.save(args.log_save_path, losses)


    # calcualting train duration
    end = time.time()
    duration = end-start
    struct_time = time.strftime("%H:%M:%S",time.gmtime(duration))

    print(f"Training finished in {struct_time}")



# code initialization
if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # run training
    main(args)

