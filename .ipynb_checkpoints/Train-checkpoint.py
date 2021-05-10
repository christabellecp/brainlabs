from initialize import *


def create_loss_and_optimizer(model, learning_rate=0.001):   
    '''
    Selects loss function and optimizer
    '''
    #Loss functions
    seg_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    #huber_loss = torch.nn.SmoothL1Loss()

    
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-8, lr=learning_rate)   
    return(seg_loss, mse_loss, optimizer)


def print_hyperparams(batch_size, epochs, learning_rate):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", epochs)
    print("learning_rate=", learning_rate)
    print()
    

# Getting validation loss and accuracy
def valid_metrics(model,  dl_val, seg_loss_fn, reg_loss_fn, optim, df, device, data_dim=128):
    '''
    Calculates validation loss and accuracy
    '''
    acc_scores = []
    seg_losses = []

    adas_losses = []
    adas_corrs  = []
    adas_r2s    = []
    i=1
    with torch.no_grad():
        for (X, y_img, y_adas, filename) in dl_val:
            model.eval()
            dim1,dim2,dim3,dim4 = y_img.size()
            total_pixels        = dim1*dim2*dim3*dim4
            
            
            X       = X.view(dim1,1,dim2,dim3,dim4).to(device)
            y_img   = y_img.squeeze(1).long().to(device)
            y_adas = y_adas.unsqueeze(1).float().to(device)
            
            adas_out, seg_out = model(X)
            M = df[df.filenames.isin(filename)].M.values
            
            linear_pred = torch.Tensor(M*0.09340 -0.0957339).to('cuda:0') + adas_out.view(-1)
                

            adas_loss = reg_loss_fn(linear_pred, y_adas.squeeze(1))


            seg_loss = seg_loss_fn(seg_out, y_img)

            _, preds = torch.max(seg_out.data, 1)

            seg_losses.append(seg_loss.item())
            adas_losses.append(adas_loss.item())
            acc_scores.append(preds.eq(y_img).sum().item()/total_pixels) 

            adas_r2 = r2_score(y_adas.detach().cpu().squeeze(1), linear_pred.detach().cpu().numpy())
            adas_r2s.append(adas_r2)
            adas_corr = pearsonr(y_adas.detach().cpu().squeeze(1), linear_pred.detach().cpu().numpy())
            adas_corrs.append(adas_corr)
                
            
    val_seg_loss    = np.mean(seg_losses)
    val_adas_loss   = np.mean(adas_losses)
    val_acc         = np.mean(acc_scores)
    val_adas_r2     = np.mean(adas_r2s)
    val_adas_corr   = np.mean(adas_corrs)
    
    return val_seg_loss, val_adas_loss, val_acc, val_adas_r2, val_adas_corr 



def train(model, dl_train, dl_val, df, epochs=20, learning_rate=0.001, 
          batch_size=5, data_dim=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    print_hyperparams(batch_size, epochs, learning_rate)

    use_amp=True
    pbar = tqdm(total=epochs*len(dl_train))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    seg_loss_fn, reg_loss_fn, optim = create_loss_and_optimizer(model, learning_rate)
    
    baseline = 0.30

    
    for epoch in range(epochs):
        acc_scores = []
        seg_losses = []
        
        adas_losses = []
        adas_corrs  = []
        adas_r2s    = []

        model.train()
        for X, y_img, y_adas, filename in dl_train:

            dim1,dim2,dim3,dim4 = y_img.size()
            total_pixels        = dim1*dim2*dim3*dim4
            
            X = X.view(dim1,1,dim2,dim3,dim4).to(device)
            y_img = y_img.squeeze(1).long().to(device)
            y_adas = y_adas.unsqueeze(1).float().to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                adas_out, seg_out = model(X)
          
    #            reg_unet_loss = reg_loss_fn(reg_out.squeeze(1), y_score.squeeze(1))
    
                M = df[df.filenames.isin(filename)].M.values

                linear_pred = torch.Tensor(M*0.09340 -0.0957339).to('cuda:0') + adas_out.view(-1)
                
                adas_loss = reg_loss_fn(linear_pred, y_adas.squeeze(1))
                
            
                seg_loss = seg_loss_fn(seg_out, y_img)
                
                _, preds = torch.max(seg_out.data, 1)

                seg_losses.append(seg_loss.item())
                adas_losses.append(adas_loss.item())
                acc_scores.append(preds.eq(y_img).sum().item()/total_pixels) 
                
                adas_r2 = r2_score(y_adas.detach().cpu().squeeze(1), linear_pred.detach().cpu().numpy())
                adas_r2s.append(adas_r2)
                adas_corr = pearsonr(y_adas.detach().cpu().squeeze(1), linear_pred.detach().cpu().numpy())
                adas_corrs.append(adas_corr)
            
            optim.zero_grad()
            loss = seg_loss + adas_loss  
            scaler.scale(loss).backward()         
            scaler.step(optim)                
            scaler.update() 
            pbar.update()

        train_seg_loss    = np.mean(seg_losses)
        train_adas_loss   = np.mean(adas_losses)
        train_acc         = np.mean(acc_scores)
        train_adas_r2     = np.mean(adas_r2s)
        train_adas_corr   = np.mean(adas_corrs)

        val_seg_loss, val_adas_loss, val_acc, val_adas_r2,val_adas_corr = valid_metrics(model, dl_val, seg_loss_fn,reg_loss_fn, optim, df, device, data_dim)
        

        
        if val_adas_r2>baseline:
            path = f'/home/dma14/brainlabs_prep/models/regvR2_{str(np.round(val_adas_r2, 4))}_tR2_{str(np.round(train_adas_r2, 4))}.pt'
            torch.save(model.state_dict(),path)
            
    
            baseline = val_adas_r2
            
        print(f"Epoch: {epoch+1} \n\
                Train Seg Loss     : {np.round(train_seg_loss, 4)}\n\
                Train Seg Acc      : {np.round(train_acc, 4)}\n\
                Valid Seg Loss     : {np.round(val_seg_loss, 4)}\n\
                Valid Seg Acc      : {np.round(val_acc, 4)}\n\n\
                Train ADAS Loss    : {np.round(train_adas_loss, 4)}\n\
                Train ADAS Pearson : {np.round(train_adas_corr, 4)}\n\
                Train ADAS R2_score: {np.round(train_adas_r2, 4)}\n\
                Valid ADAS Loss    : {np.round(val_adas_loss, 4)}\n\
                Valid ADAS Pearson : {np.round(val_adas_corr, 4)}\n\
                Valid ADAS R2      : {np.round(val_adas_r2, 4)}\n\n")

        
    return model, optim


def show_test_accuracy(nums, model, dl_test, batch_size=10, 
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    use_amp = True
    model.eval()
    batch_losses = []
    total = 0
    correct = 0
    total_loss = 0
    i=0
    nums=1
    for x, y, y_score, filenames in dl_test:
        with torch.no_grad(): 
            
            y = y.squeeze(1).long().cuda()
            dim1,dim2,dim3,dim4 = y.size() #CHANGED
            x = x.view(dim1,1,dim2,dim3,dim4).cuda()
            total += dim1*dim2*dim3*dim4  


            with torch.cuda.amp.autocast(enabled=use_amp): 

                total += y.shape[0]
                reg_out, y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                batch_losses.append(loss.item())
                pred = torch.max(y_hat, 1)[1]
                correct += (pred == y).float().sum().item()   

                if i < nums:
                    slice_idx = random.randint(40,100)
                    fig, ax = plt.subplots(3,3, figsize=(10,10))
                    fig.set_facecolor("black")
                    ax=ax.flatten()
                    sag_record = [x[i][0,:,:,slice_idx], y[i][:,:,slice_idx], pred[i][:,:,slice_idx]]
                    hor_record = [x[i][0,:,slice_idx,:], y[i][:,slice_idx,:], pred[i][:,slice_idx,:]]
                    cor_record = [x[i][0,slice_idx,:,:], y[i][slice_idx,:,:], pred[i][slice_idx,:,:]]

                    for idx in range(0,3):
                        ax[idx].set_facecolor('black')
                        ax[idx].imshow((sag_record[idx]).cpu().numpy().reshape(128,128))
                        ax[idx+3].set_facecolor('black')
                        ax[idx+3].imshow((hor_record[idx]).cpu().numpy().reshape(128,128))
                        ax[idx+6].set_facecolor('black')
                        ax[idx+6].imshow((cor_record[idx]).cpu().numpy().reshape(128,128))
                        
                    i += 1
    print(f'\nCorrect predictions percentage is: {np.round((correct*100/total), 4)}')
    
    
def train_epochs(model_path, filename, epochs, lr, batch_size, num_workers):
    
    
    X_train, X_val, y_train, y_val, X_path, y_seg_path, y_cog_path, df = initialize_data()
    ds_train, ds_val, dl_train, dl_val = get_ds_dl(batch_size=batch_size, num_workers=num_workers)
    stdout_backup = sys.stdout
    model = Smaller_RUNet3D(1,4).cuda()
    device = torch.device("cuda")
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f'/home/dma14/brainlabs_prep/models/{model_path}'))
    
    with open(f'model_logs/{filename}', 'x') as f:
        sys.stdout = f
        model, optim = train(model, None, dl_train, dl_val, df, epochs=epochs,
                             learning_rate=lr, batch_size=batch_size, device=device)
    torch.cuda.empty_cache() 
    sys.stdout = stdout_backup