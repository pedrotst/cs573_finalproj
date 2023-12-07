import os
import time
import torch
import models
import utils
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import AgglomerativeClustering


from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

import seaborn as sns  # For visualizing the confusion matrix
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.cluster import hierarchy

import pdb

class Gan:
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, optimizer_C, opt):
        # make dictionary story the losses
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_C = optimizer_C
        self.cuda = torch.cuda.is_available()
        self.opt = opt
        self.noise_intensity = 0

        self.adversarial_loss = torch.nn.BCELoss()

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
    
    def logging(self, epoch, dataloader, d_loss_epoch, g_loss_epoch, c_loss_1_epoch, c_loss_2_epoch, classification_accuracy, interval, plot_imgs, all_targets, all_predictions):
        
        num_batches = len(dataloader)
        print ("[Epoch %d/%d] [D loss: %f] [G loss: %f] [C loss 1: %f] [C loss 2: %f] [Acc: %.2f%%] \t Time Interv: %f"
                    % (epoch, self.opt.n_epochs, d_loss_epoch/num_batches, g_loss_epoch/num_batches, 
                        c_loss_1_epoch/num_batches, c_loss_2_epoch/num_batches, classification_accuracy, interval))
        
        true_labels, cluster_preds = utils.get_cluster(dataloader, self.discriminator)
        true_labels = true_labels.cpu().numpy().astype(np.int32)
        cluster_preds = cluster_preds.cpu().numpy().astype(np.int32)
        nmi_result = nmi(true_labels, cluster_preds)
        acc_result = utils.clustering_acc(true_labels, cluster_preds)

        print('Clustering metrics:', flush=True)
        print(f'NMI:{nmi_result}', flush=True)
        print(f'ACC:{acc_result}', flush=True)

        # create a directory called plot to store all the confusion matrices and generated images
        logs_path = os.path.join(os.getcwd(),self.opt.logs_path)
        # if not (os.path.exists(logs_path)):
        os.makedirs(logs_path, exist_ok=True)
        
        plot_path = os.path.join(os.getcwd(),self.opt.logs_path,'plot')
        # if not (os.path.exists(plot_path)):
        os.makedirs(logs_path, exist_ok=True)

        img_grid = make_grid(plot_imgs[:self.opt.n_paths_G*10].cpu(), nrow=10, normalize=True)
        utils.show(img_grid, self.opt)
        # save_image(img_grid, plot_path + '.jpg')
        
        if not (os.path.exists(os.path.join(plot_path,f'epoch{(epoch)}'))):
            os.makedirs(os.path.join(plot_path,f'epoch{(epoch)}'))
        
        plt.savefig(os.path.join(plot_path,f'epoch{(epoch)}', 'generated_imgs.png'), )
        cm = confusion_matrix(all_targets, all_predictions)/(100*len(dataloader))

        # Compute and print confusion matrix at the end of each epoch
        cm = confusion_matrix(all_targets, all_predictions)/(self.opt.batch_size_g*len(dataloader))
        cm_rounded = np.around(cm, decimals=2)
        np.save(os.path.join(os.getcwd(), self.opt.logs_path, 'plot',f'epoch{(epoch)}', 'confusion_matrix.npy'), cm_rounded)
        fig, ax = plt.subplots(figsize=(self.opt.n_paths_G//2, self.opt.n_paths_G//2))
        # sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        sns.heatmap(cm_rounded, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'viridis')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.savefig(os.path.join(plot_path,f'epoch{(epoch)}', 'confusion_matrix.png'))


        # Agglomerative clustering from 50 to 10 clusters
        min_num_clusters = 9
        nmi_list =[]
        acc_list =[]
        for num_clusters in range(self.opt.n_paths_G, min_num_clusters,-1):
            model = AgglomerativeClustering(linkage='single', n_clusters=num_clusters, metric='euclidean', compute_distances=True)
            labels = model.fit_predict(cm_rounded)
            new_labels = np.zeros(cluster_preds.shape)
            # Update new_labels based on cluster assignments
            for cluster_idx, cluster_label in enumerate(np.unique(cluster_preds)):
                cluster_mask = cluster_preds == cluster_label
                new_labels[cluster_mask] = labels[cluster_idx]
            
            nmi_list.append((nmi(true_labels, new_labels)))
            acc_list.append(utils.clustering_acc(true_labels, new_labels.astype(np.int64)))

        print(f"Acc for multiple agglo levels {acc_list}")
        print(f"Nmi for multiple agglo levels {nmi_list}")
        np.save(os.path.join(os.getcwd(), self.opt.logs_path, 'plot',f'epoch{(epoch)}', 'nmi_scores.npy'), np.array(nmi_list))
        np.save(os.path.join(os.getcwd(), self.opt.logs_path, 'plot',f'epoch{(epoch)}', 'acc_scores.npy'), np.array(acc_list))


    def train(self, dataloader):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        for epoch in range(self.opt.n_epochs):

            start = time.time()
            all_targets = []
            all_predictions = []
            
            epoch_start = 0
            
            g_loss_epoch = 0
            d_loss_epoch = 0
            c_loss_1_epoch = 0
            c_loss_2_epoch = 0
            
            correct_predictions = 0
            total_predictions = 0
            
            self.noise_intensity = max(self.opt.init_noise - (epoch*self.opt.noise_decay), 0)

            num_batches = len(dataloader)
            
            for i, (imgs, _) in enumerate(dataloader):
                
                valid_d = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                valid_g = Variable(Tensor(self.opt.batch_size_g, 1).fill_(1.0), requires_grad=False)
                fake_d = Variable(Tensor(self.opt.batch_size_g, 1).fill_(0.0), requires_grad=False)

                real_imgs = Variable(imgs.type(Tensor))
                real_imgs = utils.add_noise(real_imgs, self.noise_intensity)
                
                # -----------------------------------------------------
                #  Train Generator 
                # -----------------------------------------------------

                self.optimizer_G.zero_grad()

                z = Variable(Tensor(np.random.normal(0, 1, (self.opt.batch_size_g, self.opt.latent_dim))))
                g_loss = 0 
                c_loss_1 = 0 
                
                for k in range(self.opt.n_paths_G):

                    gen_imgs = self.generator.paths[k](z)
                    gen_imgs = utils.add_noise(gen_imgs, self.noise_intensity)
                    # if("cnn" in self.generator.architecture):
                        # noise = torch.zeros_like(gen_imgs).cuda()
                        # noise = noise + (0.1**0.5)*torch.randn(noise.shape).cuda()
                        # gen_imgs = gen_imgs + noise
                    
                    validity, classifier = self.discriminator(gen_imgs)
                    
                    g_loss += self.adversarial_loss(validity, valid_g)

                    target = Variable(Tensor(self.opt.batch_size_g).fill_(k), requires_grad=False)
                    target = target.type(torch.cuda.LongTensor)
                    
                    c_loss_1 += F.nll_loss(classifier, target) * self.opt.classifier_para_g

                    _, predicted = torch.max(classifier.data, 1)
                    total_predictions += target.size(0)
                    correct_predictions += (predicted == target).sum().item()
            
                    # Collect all targets and predictions for confusion matrix
                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    # train_on_batch
               
                
                g_loss_epoch += g_loss
                c_loss_1_epoch += c_loss_1       
                g_loss = g_loss + c_loss_1
                
                g_loss.backward()
                self.optimizer_G.step()
                
                # ------------------------------------------------------------------------
                #  Train Discriminator and Classifier (WE CALL THIS STAGE PHASE 2)
                # ------------------------------------------------------------------------

                self.optimizer_D.zero_grad()
                self.optimizer_C.zero_grad()

                d_loss = 0 #adversarial loss for discrimination to be applied to the discriminator
                c_loss_2 = 0 #classification loss to be applied to the classifier
                
                #loss of the discriminator with real images
                validity, classifier = self.discriminator(real_imgs)
                
                real_loss = self.adversarial_loss(validity, valid_d)
                
                temp = [] #variable to store images for plot
                
                #again we iterate over each of the generators
                for k in range(self.opt.n_paths_G):

                    # generates a batch of images with generator k
                    gen_imgs = self.generator.paths[k](z).view(self.opt.batch_size_g, *(self.opt.img_shape))
                    gen_imgs_no_noise = torch.clone(gen_imgs).detach()
                    gen_imgs = utils.add_noise(gen_imgs, self.noise_intensity)
                    
                    # if("cnn" in self.generator.architecture):
                        # noise = torch.zeros_like(gen_imgs).cuda()
                        # noise = noise + (0.1**0.5)*torch.randn(noise.shape).cuda()
                        # gen_imgs = gen_imgs + noise

                    temp.append(gen_imgs_no_noise[0:10, :])

                    #again we measure the outputs of the discriminator and the classifier 
                    validity, classifier = self.discriminator(gen_imgs.detach())
                    
                    #loss of the discriminator with fake images
                    fake_loss = self.adversarial_loss(validity, fake_d)
                    
                    #sums up the fake and true losses
                    d_loss += (real_loss + fake_loss) / 2

                    #again we calculate the loss of the classifier. Note that it is calculated...
                    #...in exactly the same way as before, but only after the...
                    #...generator has already had its weights updated in PHASE 1.
                    #reminder: opt.classifier_para is just a multiplicative constant of the loss
                    target = Variable(Tensor(self.opt.batch_size_g).fill_(k), requires_grad=False)
                    target = target.type(torch.cuda.LongTensor)
                    c_loss_2 += F.nll_loss(classifier, target) * self.opt.classifier_para
                
                #images for plot
                plot_imgs = torch.cat(temp, dim=0)


                #again, we accumulate the losses to visualize the total for the epoch
                d_loss_epoch += d_loss
                c_loss_2_epoch += c_loss_2
                
                #we add both losses into the same variable 
                d_loss = d_loss + c_loss_2
                #jointly apply the gradients to the layers of the discriminator and classifier
                d_loss.backward()
                
                self.optimizer_D.step()
                self.optimizer_C.step()
            
            #printing training information
            interval = time.time() - start
            
            classification_accuracy = 100.0 * correct_predictions / total_predictions
            
            # if (epoch < 20 and epoch % 5 == 0) or (epoch > 20 and epoch % 20 == 0):
            if (epoch % self.opt.sample_interval == 0):
                self.logging(epoch, dataloader, d_loss_epoch.item(), g_loss_epoch.item(), c_loss_1_epoch.item(), c_loss_2_epoch.item(), classification_accuracy, interval, plot_imgs, all_targets, all_predictions)

                
                
                
               

                
                
                
                
