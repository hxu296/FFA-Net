

# --- Imports --- #
import torch.utils.data as data
from PIL import Image,ImageFile
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import random
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class TrainDataUnreal(data.Dataset):
    def __init__(self,dataset_name, crop_size, train_data_dir,):
        super().__init__()             
        ### folder structure related
        self.root_dir = train_data_dir# keep this for train/val/test split
        self.img_names = []
        self.image_depth_names = []
        foler0_names = os.listdir(self.root_dir)

        for foler0_name in foler0_names:
            if len(self.focal_length) != 0:
              foler0_name.split('_')[-1] 
              if foler0_name.split('_')[-1] not in self.focal_length:
                continue
            foler0_name_path = os.path.join(self.root_dir,foler0_name)
            print('foler0_name_path',foler0_name_path)
            img_names = sorted(os.listdir(os.path.join(foler0_name_path,'ColorImage'))) # name of all images 
            depth_names = sorted(os.listdir(os.path.join(foler0_name_path,'DepthImage'))) # name of all images 
            for img_name,depth_name in zip(img_names,depth_names):		
              depth_path = os.path.join(foler0_name_path,"DepthImage",depth_name)
              img_path = os.path.join(foler0_name_path,"ColorImage",img_name)
              self.image_depth_names.append([img_path,depth_path])
        print(">>> Total number of training examples is",len(self.image_depth_names),len(sorted(os.listdir(os.path.join(foler0_name_path,'DepthImage')))))

        ## set up resize if styled image has been resized

        self.crop_size = crop_size

        self.size_w = crop_size[0]
        self.size_h = crop_size[1]

    def cv2PIL(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

    def convert(self, im, t, A):
        res = np.empty(im.shape, im.dtype)
        for ind in range(0,3):
            res[:,:,ind] = im[:,:,ind] * t + A[0,0] * (1 - t)
        return res

    def asm_haze_gen(self,im,gt_depth,sky_mask,beta0=0.6,beta1=1.8,A0=0.7,A1=1,beta_sky0=0.3,beta_sky1=0.9):	
        beta = random.uniform(beta0,beta1)
        beta_sky = random.uniform(beta_sky0,beta_sky1)
        A = np.random.uniform(A0,A1,size=(1,3))
        t = np.exp(-beta * gt_depth)           

        ## use thinner fog for the sky region
        t[sky_mask] = np.exp(-(beta_sky)*gt_depth[sky_mask])

        haze_img = self.convert(im,t,A)
        
        return {
            'hazy_img': haze_img,
            'A': A,
            'beta': beta,
            'beta_sky':beta_sky
        }
    
    def asm_haze_gen_fixed_val(self,im,gt_depth,beta,A):	
        t = np.exp(-beta * gt_depth)
        haze_img = self.convert(im,t,A)
        
        return {
            'hazy_img': haze_img,
            'A': A,
            'beta': beta
        }
    
    def depth_map_conversion(self,depth_map):
        """
        GTA data sets sky region to inf. we set inf to 255 and map other regions to [0,255].
        This is a bit undersiable as sky region may have the same depth values with other regions that is far but not sky.
        """
        mask = depth_map == 255
        depth_map =  (depth_map - np.min(depth_map))/(np.max(depth_map)-np.min(depth_map))
        return depth_map,mask

    def get_images(self, index):

        #### read in the depth map & clean images
        gt_name,gt_depth_name = self.image_depth_names[index]
        gt_depth = cv2.imread(gt_depth_name,0)
        target_img = cv2.imread(os.path.join(self.root_dir, gt_name)) # clear imgs


        ### construct hazy images
        target_img = target_img.astype('float32') / 255.0 # conver to range in (0,1) as ASM implementation assumes in that range
        gt_depth,sky_mask = self.depth_map_conversion(gt_depth)
    
        source_dic = self.asm_haze_gen(target_img,gt_depth,sky_mask,beta0=self.beta0,beta1=self.beta1)
        source_img,A,beta = source_dic['hazy_img'],source_dic['A'],source_dic['beta']


        ### convert & do resizing
        haze = source_img * 255.0 ## this part may have problem
        haze = self.cv2PIL(haze)
        haze = haze.resize((self.size_w, self.size_h))

        clear = target_img * 255.0
        clear = clear.resize((self.size_w, self.size_h))
        clear = clear.resize((self.size_w, self.size_h)) 

        haze,gt=self.augData(haze.convert("RGB") ,clear.convert("RGB") )

        # --- Check the channel is 3 or not --- # 
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3: 
            #print(gt_name)
            raise Exception('Bad image channel: {}'.format(gt_name))
        return haze, gt
    

    def augData(self,data,target):
        #if self.train:
        if 1: 
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target

    def __getitem__(self, index):
        res = self.get_images(index)
        return res 

    def __len__(self):
        return len(self.haze_names)

