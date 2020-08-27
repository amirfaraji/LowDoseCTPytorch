import sys
import copy
import time
import numpy as np
import torch

sys.path.append('../src')

from torchsummary import summary
from utils.CustomBar import CustomBar, Bar
from utils.Visualize import visualize
from dival.measure import PSNR, SSIM



class ModelClient(object):
    
    def __init__(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass



class ReconstructionModelClient(ModelClient):

    def __init__(self, path: str='', summary: bool=False):
        super(ModelClient, self).__init__()
        self.path = './src/weights/state_dict_model.pt'
        self.summary = summary
        

    def train(self, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs=200):
        if self.summary:
            print(summary(model.cuda(), train_dataset.shape))

        since = time.time()
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000000.
        self.history = {'train':[None]*epochs, 'val':[None]*epochs}

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 40)


            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = train_dataloader
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = val_dataloader

                bar = CustomBar(f'Processing {phase} set', max=len(dataloader))

                running_loss = [0.0]*len(dataloader)
                running_ssim = [0.0]*len(dataloader)
                running_psnr = [0.0]*len(dataloader)

                for i, batch in enumerate(dataloader):
                    inputs = batch['observation']
                    labels = batch['gt']
                    # inputs = self.get_patches(inputs, 2, 2)
                    # labels = self.get_patches(labels, 2, 2)
                    
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    outs, targets = outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
                    
                    running_loss[i] = loss.item()
                    running_ssim[i] = self.get_mean_batch_metric(outs, targets, SSIM)  
                    running_psnr[i] = self.get_mean_batch_metric(outs, targets, PSNR)
                    bar.loss = running_loss[i]
                    bar.ssim = running_ssim[i]
                    bar.psnr = running_psnr[i]
                    bar.next()

                if phase == 'train' and scheduler:
                    scheduler.step(loss)

                bar.finish()

                epoch_loss = np.mean(running_loss)
                epoch_ssim = np.mean(running_ssim)
                epoch_psnr = np.mean(running_psnr)
                
                print('{} Loss: {:.4f} SSIM: {:.4f} PSNR: {:.4f}'.format(
                    phase, epoch_loss, epoch_ssim, epoch_psnr))
                
                self.history[phase][epoch] = epoch_loss

                if phase == 'val' and epoch_loss < best_loss:
                    print("Saving model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best_model_wts.pt')


            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        model.load_state_dict(best_model_wts)
        return model

    def predict(self, model, loader, device):

        # TODO: Implement batch prediction

        # model.eval()
        # outputs = np.zeros((3553,1,180,180))

        # bar = Bar(f'Processing test set', max=len(loader), suffix='%(index)d / %(max)d - %(eta)d secs')

        # for i, batch in enumerate(loader):
        #     inputs = batch['observation']
        #     targets = batch['gt']
        #     inputs = inputs.to(device, dtype=torch.float)
        #     outs = model(inputs)
        #     outputs[i*len(inputs):(i+1)*len(inputs),:,:,:] = outs.detach().cpu().numpy()
        #     bar.next()

        # bar.finish()

        # return np.squeeze(outputs)
        pass

    def single_predict(self, model, img, device):
        model.eval()
        inputs = img
        inputs = inputs.to(device, dtype=torch.float)
        outputs = model(inputs)
        outputs= outputs.cpu().data.numpy()
        return outputs

    def batch_evaluation(self, model, loader, device):
        model.eval()
        ssim_val = [0]*len(loader)
        psnr_val = [0]*len(loader)

        bar = Bar(f'Processing test set', max=len(loader), suffix='%(index)d / %(max)d - %(eta)d secs')

        for i, batch in enumerate(loader):
            inputs = batch['observation']
            targets = batch['gt']
            # inputs = self.get_patches(inputs, 2, 2)
            inputs = inputs.to(device, dtype=torch.float)
            outputs = model(inputs)
            # outputs = self.unfold_patches_from_4_patches(targets.shape, outputs)

            outs = outputs.detach().cpu().numpy()
            ground_truths = targets.detach().cpu().numpy()


            ssim_val[i] = self.get_mean_batch_metric(outs, ground_truths, SSIM)  
            psnr_val[i] = self. get_mean_batch_metric(outs, ground_truths, PSNR)
            bar.next()
            # visualize(outs[0,0,:,:], ground_truths[0,0,:,:])
        bar.finish()

        return np.mean(ssim_val), np.mean(psnr_val)
        
    def get_mean_batch_metric(self, outs, targets, metric):
        metric_ = [0]*len(outs)
        for i in range(len(outs)):
            metric_[i] = metric(outs[i,0,:,:], targets[i,0,:,:])
        
        return np.mean(metric_)

    
    def get_patches(self, imgs, num_width_patch, num_height_patch):
        patches = torch.zeros(
            (imgs.shape[0]*num_width_patch*num_height_patch, imgs.shape[1], imgs.shape[2]//num_width_patch, imgs.shape[3]//num_height_patch), 
            dtype=torch.float32
        )
        
        count = 0
        for img in range(len(imgs)):
            for i in range(num_width_patch):
                for j in range(num_width_patch):
                    patches[count,:,:,:] = imgs[img,:,
                                            i*(imgs.shape[2]//num_width_patch):(i+1)*(imgs.shape[2]//num_width_patch), 
                                            j*(imgs.shape[3]//num_height_patch):(j+1)*(imgs.shape[3]//num_height_patch)]
                    count +=1
        return patches

    def unfold_patches_from_4_patches(self, size, imgs):
        full_img = torch.zeros(
            (size), 
            dtype=torch.float32
        )

        for i in range(len(full_img)):
                    full_img[i,:,0:181,0:181] = imgs[i,:,:,:]
                    full_img[i,:,0:181,181:] = imgs[i+1,:,:,:]
                    full_img[i,:,181:,0:181] = imgs[i+2,:,:,:]
                    full_img[i,:,181:,181:] = imgs[i+3,:,:,:]

        return full_img
    
