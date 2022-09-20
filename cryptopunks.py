import streamlit as st
import numpy as np
import torch
from torch import device
from torchvision.utils import save_image
import torch.nn as nn
import random

latent_size= 50

class Generator_64x64(nn.Module):
    def __init__(self):
        super(Generator_64x64, self).__init__()
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size+5, 512, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            #nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
                
        self.upsample3 = nn.Sequential(
            #nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)            
        )

        self.upsample4 = nn.Sequential(
            #nn.ConvTranspose2d(128, 24, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(128, 24, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True)    
        )

        # self.upsample6 = nn.Sequential(
        #     nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(True)    
        # )

        self.upsample5 = nn.Sequential(
            #nn.ConvTranspose2d(24, 3, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, class_vec):
        x = torch.cat([x,class_vec], 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        #x = self.upsample6(x)
        x = self.upsample5(x)
        return x

class Generator_28x28(nn.Module):
    def __init__(self):
        super(Generator_28x28, self).__init__()
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size+5, 512, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            #nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
                
        self.upsample3 = nn.Sequential(
            #nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)            
        )
        self.upsample4 = nn.Sequential(
            #nn.ConvTranspose2d(128, 24, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(128, 24, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True)    
        )

        # self.upsample6 = nn.Sequential(
        #     nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=0, bias=False),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(True)    
        # )

        self.upsample5 = nn.Sequential(
            #nn.ConvTranspose2d(24, 3, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(24, 3, kernel_size=2, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, class_vec):
        x = torch.cat([x,class_vec], 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        #x = self.upsample6(x)
        x = self.upsample5(x)
        return x

def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

def main():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Тут генерируются криптопанки :alien:')
        choice_type = st.radio('Какого хотите?', options=['женщину', 'мужчину', 'пришельца', 'зомби', 'обезьяну'], index=1, on_change=None, disabled =False)
        choice_size = st.radio('Какого размера?', options=['28х28', '64х64'], index=1, on_change=None, disabled =False)
        num_return_sequences = st.slider('Сколько панков?', min_value=1, max_value=5, value=1, step=1, format=None, 
        key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
        result = st.button('Поехали!')
    with col2:
        # st.image(np.random.randint(0, 255, size=(50, 50, 3)))
        # st.image('images/1.png', width=50)
        # st.image('images/2.png', width=50)
        # st.image('images/3.png', width=50)
        images = ['images/3.png']
        st.image(images, width  = 100, caption=["one of them"] * len(images))

    choice_dict = {'женщину':3, 'мужчину':1, 'пришельца':2, 'зомби':4, 'обезьяну':0}
    
    if result:
        generator = Generator_28x28()
        # generator = Generator_64x64()
        images =[]
        for i in range(num_return_sequences):
            i  = random.randint(35,58)## выбор весов из эпох от 25 до 59
            name = 'weights/punk_generator_ws24_'+str(i)+'.pt'
            generator.load_state_dict(torch.load(name, map_location=torch.device('cpu')))
            label = torch.Tensor([choice_dict[choice_type]])
            target = nn.functional.one_hot(label.unsqueeze(1).unsqueeze(1).to(torch.int64), 5).permute(0,3,1,2).float()
            target=(target*0.96).to('cpu')
            generator.eval()
            noise = torch.randn(1, latent_size, 1, 1).to('cpu') 
            tensor = generator(noise, target)
            save_image(denorm(tensor), str(i)+'.jpg', normalize=True)
            images.append(str(i)+'.jpg')
        with col2:
            # st.image(images, width=60)
            st.image('images/image-0038.png', use_column_width=True)

if __name__ == '__main__':
         main()