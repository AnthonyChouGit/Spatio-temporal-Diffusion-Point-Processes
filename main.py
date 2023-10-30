from utils import get_args, setup_init, data_loader, enc_batch, LR_warmup
import setproctitle
from torch.utils.tensorboard import SummaryWriter
from DSTPP import GaussianDiffusion_ST, Transformer_ST, Model_all, ST_Diffusion
from torch.optim import AdamW
import torch
import os
import numpy as np


if __name__ == '__main__':
    opt = get_args()
    device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

    if opt.dataset == 'HawkesGMM':
        opt.dim=1

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)
    setup_init(opt.seed)
    setproctitle.setproctitle("Model-Training")
    print('dataset:{}'.format(opt.dataset))

    # Specify a directory for logging data 
    logdir = "./logs/{}_timesteps_{}".format( opt.dataset,  opt.timesteps)
    model_path = './ModelSave/dataset_{}_timesteps_{}/'.format(opt.dataset, opt.timesteps) 

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)
    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    model= ST_Diffusion(
        n_steps=opt.timesteps,
        dim=1+opt.dim,
        condition = True,
        cond_dim=64
    ).to(device)

    diffusion = GaussianDiffusion_ST(
        model,
        loss_type = opt.loss_type,
        seq_length = 1+opt.dim,
        timesteps = opt.timesteps,
        sampling_timesteps = opt.samplingsteps,
        objective = opt.objective,
        beta_schedule = opt.beta_schedule
    ).to(device)

    transformer = Transformer_ST(
        d_model=64,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim = opt.dim,
        CosSin = True
    ).to(device)

    Model = Model_all(transformer,diffusion)

    trainloader, testloader, valloader, (MAX,MIN) = data_loader(opt.dataset, opt.batch_size, opt.dim)

    warmup_steps = 5

    optimizer = AdamW(Model.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for epoch in range(opt.total_epochs):
        print(f'epoch:{epoch}')
        if epoch % 10 == 0:
            print('Evaluate...')
            with torch.no_grad():
                Model.eval()

                # Validation set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = enc_batch(batch, Model.transformer, opt.dim, device)
                    # sampled_seq: (batch_size, channels, seq_length) number of channels is set to 1 by hard coding
                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    
                    real = (event_time_non_mask[:,0,:].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
                    gen = (sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
                    assert real.shape==gen.shape
                    mae_temporal += torch.abs(real-gen).sum().item()
                    rmse_temporal += ((real-gen)**2).sum().item()
                    real = event_loc_non_mask[:,0,:].detach().cpu()
                    assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    real = (real + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    gen = sampled_seq[:,0,-opt.dim:].detach().cpu()
                    gen = (gen + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    assert real.shape==gen.shape
                    mae_spatial += torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()
                    total_num += gen.shape[0]

                    assert gen.shape[0] == event_time_non_mask.shape[0]
                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 10:
                        break
                else:
                    early_stop = 0
                torch.save(Model.state_dict(), model_path+'model_{}.pkl'.format(epoch))

                min_loss_test = min(min_loss_test, loss_test_all)

                writer.add_scalar(tag='Evaluation/loss_val',scalar_value=loss_test_all/total_num,global_step=epoch)

                writer.add_scalar(tag='Evaluation/NLL_val',scalar_value=vb_test_all/total_num,global_step=epoch)
                writer.add_scalar(tag='Evaluation/NLL_temporal_val',scalar_value=vb_test_temporal_all/total_num,global_step=epoch)
                writer.add_scalar(tag='Evaluation/NLL_spatial_val',scalar_value=vb_test_spatial_all/total_num,global_step=epoch)

                writer.add_scalar(tag='Evaluation/mae_temporal_val',scalar_value=mae_temporal/total_num,global_step=epoch)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val',scalar_value=np.sqrt(rmse_temporal/total_num),global_step=epoch)
                writer.add_scalar(tag='Evaluation/distance_spatial_val',scalar_value=mae_spatial/total_num,global_step=epoch)
                # test set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in testloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = enc_batch(batch, Model.transformer, opt.dim, device)

                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    total_num += gen.shape[0]
                writer.add_scalar(tag='Evaluation/loss_test',scalar_value=loss_test_all/total_num,global_step=epoch)

                writer.add_scalar(tag='Evaluation/NLL_test',scalar_value=vb_test_all/total_num,global_step=epoch)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test',scalar_value=vb_test_temporal_all/total_num,global_step=epoch)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test',scalar_value=vb_test_spatial_all/total_num,global_step=epoch)
        if epoch < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, epoch)
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3- (1e-3 - 5e-5)*(epoch-warmup_steps)/opt.total_epochs
                param_group["lr"] = lr

        writer.add_scalar(tag='Statistics/lr',scalar_value=lr,global_step=epoch)

        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:
            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = enc_batch(batch, Model.transformer, opt.dim, device)
            loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()

            loss_all += loss.item() * event_time_non_mask.shape[0]
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            writer.add_scalar(tag='Training/loss_step',scalar_value=loss.item(),global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step() 
            
            step += 1

            total_num += event_time_non_mask.shape[0]
        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch',scalar_value=loss_all/total_num,global_step=epoch)

        writer.add_scalar(tag='Training/NLL_epoch',scalar_value=vb_all/total_num,global_step=epoch)
        writer.add_scalar(tag='Training/NLL_temporal_epoch',scalar_value=vb_temporal_all/total_num,global_step=epoch)
        writer.add_scalar(tag='Training/NLL_spatial_epoch',scalar_value=vb_spatial_all/total_num,global_step=epoch)
