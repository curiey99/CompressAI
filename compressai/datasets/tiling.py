import numpy as np
import torch

def feature_rearrange(feature): ## 256, 4, 4 ->  1, 64, 64
    h,w = feature.shape[1],feature.shape[2]
    featuremap = np.zeros((16*h,16*w))
    for i in range(16):
      for j in range(16):
        c_num = i*16+j
        featuremap[i*h:(i+1)*h,j*w:(j+1)*w] = feature[c_num,:,:]
    return featuremap

def feature_rearrange_torch(feature): ## 256, 4, 4 ->  1, 64, 64
    h,w = feature.shape[1],feature.shape[2]
    featuremap = torch.zeros((16*h,16*w))
    for i in range(16):
      for j in range(16):
        c_num = i*16+j
        featuremap[i*h:(i+1)*h,j*w:(j+1)*w] = feature[c_num,:,:]
    return featuremap

def channel_rearrange(feature):  ## 64, 64 -> 256, 4, 4
    h,w = int(feature.shape[0]/16),int(feature.shape[1]/16)
    featurechannel = np.zeros((256,h,w))
    for i in range(16):
      for j in range(16):
        c_num = i*16+j
        featurechannel[c_num,:,:] = feature[i*h:(i+1)*h,j*w:(j+1)*w]
    featurechannel = np.float32(featurechannel)
    return featurechannel

def channel_half_rearrange(feature):  ## 64, 64 -> 4, 32, 32
    h,w = int(feature.shape[0]/2),int(feature.shape[1]/2) # 2, 2
    featurechannel = np.zeros((4,h,w)) # 4
    for i in range(2):
      for j in range(2):
        c_num = i*2+j
        featurechannel[c_num,:,:] = feature[i*h:(i+1)*h,j*w:(j+1)*w]
    featurechannel = np.float32(featurechannel)
    return featurechannel

def channel_half_rearrange_torch(feature):  ## 64, 64 -> 4, 32, 32
    h,w = int(feature.shape[0]/2),int(feature.shape[1]/2) # 2, 2
    featurechannel = torch.zeros((4,h,w)) # 4
    for i in range(2):
      for j in range(2):
        c_num = i*2+j
        featurechannel[c_num,:,:] = feature[i*h:(i+1)*h,j*w:(j+1)*w]
    # featurechannel = np.float32(featurechannel)
    return featurechannel


def channel_final_rearrange(feature):  ## 4, 32, 32 -> 256, 4, 4
    h,w = int(feature.shape[1]/8),int(feature.shape[2]/8) # 2, 2
    
    featurechannel = np.zeros((256,h,w)) # 4
    for c in range(64):
        featurechannel[c, :, :] = feature[(c%32)//8, (c%4)//4*h:(c//8)*h+h, (c%2)*w:(c%2)*w+w]

    return featurechannel

def channel_final_rearrange_256(feature):  ## 4, 2048, 2048 -> 256, 256, 256 (64x)((c%64)//16)*h+h
    h,w = int(feature.shape[1]/8),int(feature.shape[2]/8) # 2, 2   ((c%64)//8)*h:((c%64)//8)*h+h
    featurechannel = np.zeros((256,h,w)) # 4
    for c in range(256):
        # print(c)
        featurechannel[c, :, :] = feature[(c//128)*2+(c//8)%2, ((c%128)//16)*h:((c%128)//16)*h+h, (c%8)*w:(c%8)*w+w]
        
    featurechannel = np.float32(featurechannel)
    return featurechannel

def channel_final_rearrange_256_torch(feature):  ## 4, 2048, 2048 -> 256, 256, 256 (64x)((c%64)//16)*h+h
    h,w = int(feature.shape[1]/8),int(feature.shape[2]/8) # 2, 2   ((c%64)//8)*h:((c%64)//8)*h+h
    featurechannel = torch.zeros((256,h,w)) # 4
    # print("tiling size: {}".format(feature.shape))
    for c in range(256):
        # print(c)
        featurechannel[c, :, :] = feature[(c//128)*2+(c//8)%2, ((c%128)//16)*h:((c%128)//16)*h+h, (c%8)*w:(c%8)*w+w]
        
    # featurechannel = np.float32(featurechannel)
    return featurechannel.to(torch.float32)


def tile_256_to_4(feature):
    feature = feature_rearrange(feature)
    feature = np.squeeze(feature, 0)
    feature = channel_half_rearrange(feature)
    return feature

def tile_256_to_4_torch(feature):
    feature = feature_rearrange_torch(feature).squeeze(0)
    feature = channel_half_rearrange_torch(feature)
    return feature # (4, 16h, 16w)

def feature_preprocess(feature):#,rescale_factor,quanta_bit):
    # feature = downsample3d(feature,rescale_factor)
    feature_cr = feature_rearrange(feature)
    # feature_q = normal_quanta(feature_cr,quanta_bit)
    return feature_cr


def normalize(featuremap): # (4h, 4w)
    min = np.amin(featuremap)
    fmap = featuremap - min
    stretch = 255//np.amax(fmap)
    fmap = stretch * fmap
    return fmap, stretch, min