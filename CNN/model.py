from CNN.mlp import*
import numpy as np
from torch.optim import AdamW
from CNN.tools import*
import torch.nn.functional as F
import os
from time import time

class model():
    def __init__(self,params):
        self.params = params
        self.initModel()

#initialize the model
    def initModel(self):
        self.model = MLP(self.params)
        self.optimizer = AdamW(self.model.parameters(),lr=self.params['learning rate'],amsgrad=True)
        self.criterion = nn.MSELoss()



#check convergence
    def converge(self,tr,valid):
        '''
        eveluates the NN until is satisfied the convergence
        '''
        self.err_tr_hist=[]
        self.err_val_hist=[]

        #flag converge
        self.convergence = 0
        self.epoch = 0
        t0=time()
        while (self.convergence != 1):
            model.training(self,tr)
            model.validation(self,valid)
            print('epoch=%s needed time=%s' %(self.epoch,(time()-t0)))
            print('Loss training=%s'% self.err_tr_hist[self.epoch])
            print('Loss validation=%s'% self.err_val_hist[self.epoch])
            t0=time()
            if self.epoch >= self.params['epochs']:
                self.convergence = 1
            self.epoch += 1


# training module
    def training(self,tr):
        err_his=[]
        for i,(x,y) in enumerate(tr):
            if self.params['device'] == 'cuda':
                x=x.cuda()
                y=y.cuda()
            #clear gradients
            self.optimizer.zero_grad()
            # prediction
            yhat=self.model(x.view(-1,1,self.params['size'],self.params['size']))
            #calculate loss
            loss=self.criterion(yhat,y)
            err_his.append(loss.data)
            #assignment
            loss.backward()
            #update parameters
            self.optimizer.step()
        self.err_tr_hist.append(torch.mean(torch.stack(err_his)).cpu().detach().numpy())

# validation
    def validation(self,valid):
        err_his=[]
        with torch.no_grad():
            for i, (x,y)in enumerate(valid):
                if self.params['device'] == 'cuda':
                    x=x.cuda()
                    y=y.cuda()
                yhat=self.model(x.view(-1,1,self.params['size'],self.params['size']))
                loss=self.criterion(yhat,y)
                err_his.append(loss.data)
        self.err_val_hist.append(torch.mean(torch.stack(err_his)).cpu().detach().numpy())

#convergence criteria
    def checkconvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        # if test loss increases consistently
#        if all(np.asarray(self.err_val_hist[-self.params['epochs']+1:])  > self.err_val_hist[-self.params['epochs']]):
#            self.convergence = 1

        # check if test loss is unchanging
        if abs(self.err_val_hist[-self.params['epochs']] - np.average(self.err_val_hist[-self.params['epochs']:]))/self.err_val_hist[-self.params['epochs']] < eps:
            self.convergence = 1

        # check if we have hit the epoch ceiling
        if self.epochs >= self.params['max training epochs']:
            self.convergence = 1

#save the Model
    def save_model(self):
        #os.rmdir(self.params['savedir'])
        os.mkdir(self.params['savedir'])
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, '%s/%s' %(self.params['savedir'],self.params['savename']) )

#load model
    def load_model(self):
        '''
        this only load models from the same device
        this means cpu to cpu or gpu to GPU
        to do cross devices you must use:
        model.load_state_dict(torch.load(PATH, map_location=device)) fro save on GPU and load on CPU
        for save on CPU and load in GPU:
        device = torch.device("cuda")
        model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
        model.to(device)
        '''
        checkpoint = torch.load('%s/%s' %(self.params['savedir'],self.params['savename']))
        if self.params['device'] == 'cpu':
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
        else:
            device = torch.device("cuda")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(device)
            self.model.eval()


#predictions
##predictions for cat vs dog model
    def pred_catVsdog(self,path):
        ima=new_image(path,self.params['size'])
        x=torch.Tensor(ima).view(-1,1,self.params['size'],self.params['size']).to(torch.float64)
        if self.params['device'] == 'cuda':
            y = self.model(x.to('cuda'))
        else:
            y = self.model(x)

        if torch.argmax(y) == 0:
            print('the pic is a cat')
        elif torch.argmax(y) == 1:
            print('the pic is a dog')

## predition general
    def pred(self,path):
        ima=new_image(path,self.params['size'])
        x=torch.Tensor(ima).view(-1,1,self.params['size'],self.params['size']).to(torch.float64)
        if self.params['device'] == 'cuda':
            y = self.model(x.to('cuda'))
        else:
            y = self.model(x)
    return y

#visualization loss functions
    def visual_loss(self):
        plot(self.err_tr_hist,self.err_val_hist)
