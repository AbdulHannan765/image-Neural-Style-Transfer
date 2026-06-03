import argparse  
import torch
from pathlib import Path
from utils.utils import ImageDataset,get_transforms
from torch.utils.data import DataLoader
from utils.models import Decoder,VGG_Encoder
import torch.optim as optim
from tqdm import tqdm
from utils.utils import adaptive_instance_normalization,calculate_mean_std
from torchvision.utils import save_image
def parse_argument():
    parser=argparse.ArgumentParser()
    parser.add_argument("--content_dir",type=str,default="content_data",help="location of content dataset")
    parser.add_argument("--style_dir",type=str,default="style_data",help="location of style_dataset")
    parser.add_argument("--vgg",type=str,default="utils/vgg_normalised.pth",help="location of vgg model")
    parser.add_argument("--experiment",type=str,default="experiment1",help="name of the experiment")
    parser.add_argument("--final_size",type=int,default=512,help="size ogf final image")
    parser.add_argument("--content_size",type=int,default=512,help="size of the content image")
    parser.add_argument("--style_size",type=int,default=512,help="size of the style image")
    parser.add_argument("--crop",action="store_true",help="crop image")
    
  
    parser.add_argument("--batch_size",type=int,default=16,help="batch_size")
    parser.add_argument("--lr",type=float,default=0.0001,help="learning rate for decoder")
    parser.add_argument("--lr_decay",type=float,default=5e-5,help="learnig rate decay")
    parser.add_argument("--epochs",type=int,default=2,help="no of epochs")
    parser.add_argument("--resume",type=bool,default=False,help="do you want to resume")
    parser.add_argument("--decoder_path",type=str,default=None,help="decoder path")
    parser.add_argument("--optimizer_path",type=str,default=None,help="optimizer_path path")
    parser.add_argument("--content_weight",type=float,default=1.0,help="content_weights")
    parser.add_argument("--style_weight",type=float,default=10,help="style weight")
    parser.add_argument("--epoch_show",type=int,default=1)
    parser.add_argument("--save_model",type=int,default=1)
    parser.add_argument("--experiment_path",type=str,default="experiment")



    return parser.parse_args()
def main():
    args=parse_argument()
    device="cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True
    print(device)
    save_dir=Path(args.experiment_path)/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)
    with open(save_dir /"args.txt","w") as f:
        for key,value in vars(args).items():
            f.write(f"{key} : {value}\n")
    content_transform=get_transforms(args.content_size,args.crop,args.final_size)
    style_transform=get_transforms(args.style_size,args.crop,args.final_size)
    content_dataset=ImageDataset(args.content_dir,content_transform)
    style_dataset=ImageDataset(args.style_dir,style_transform)
    content_dataloader=DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True,num_workers=2,persistent_workers=True)
    style_dataloader=DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True,num_workers=2,persistent_workers=True)
    encoder=VGG_Encoder(args.vgg)
    decoder=Decoder()
    encoder.to(device)
    decoder.to(device)
    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)
    
    scheduler=optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch : 1.0/(1.0 + args.lr_decay * epoch)

     )
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
    

    mse_loss=torch.nn.MSELoss()
    encoder.eval()
   
    for epoch in range(args.epochs):
      
        print(f"epoch {epoch}")
        running_loss=0
        conent_running_loss=0
        style_running_loss=0
        progress_bar=tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))
        for content_batch,style_batch in progress_bar:
            content_batch=content_batch.to(device)
            style_batch=style_batch.to(device)

            with torch.no_grad():
                content_features = encoder(content_batch)
                style_features = encoder(style_batch)
            t=adaptive_instance_normalization(content_features[-1],style_features[-1])
            g=decoder(t)
            g_feat=encoder(g)
            loss_C=mse_loss(g_feat[-1],t)*args.content_weight
            loss_s=0
            for g_f,s_f in zip(g_feat,style_features):
                g_mean,g_std=calculate_mean_std(g_f)
                s_mean,s_std=calculate_mean_std(s_f)
                loss_s+=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std)
            loss_si=loss_s*args.style_weight
            loss=loss_C+loss_si
            running_loss+=loss.item()
            conent_running_loss+=loss_C.item()
            style_running_loss+=loss_si.item()
            progress_bar.set_description(
                f"overall={loss.item():.4f} "
                f"content={loss_C.item():.4f} "
                f"style={loss_si.item():.4f}"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        total_loss=running_loss/len(content_dataloader)
        total_c_loss=conent_running_loss/len(content_dataloader)
        total_s_loss=style_running_loss/len(content_dataloader)
        if (epoch+1) % args.epoch_show==0:
            tqdm.write(f"epoch ={epoch+1} overall loss ={total_loss} \n content loss = {total_c_loss} \n style loss {total_s_loss}")
        if (epoch+1) % args.save_model==0:
            torch.save(decoder.state_dict(),save_dir/f"decoder {epoch+1}.pth")
            torch.save(optimizer.state_dict(),save_dir/f"optimizer {epoch+1}.pth")
            with torch.no_grad():
                output=torch.cat([content_batch,style_batch,g],dim=0)
                save_image(output,save_dir/f"images {epoch+1}.jpg",nrow=args.batch_size)
if __name__ == "__main__":
    main()
                


