import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from tqdm import tqdm
import pylab
import mpl_toolkits.mplot3d as Axis3D


class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(10, 1, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten()

        )
        self.compress = nn.Sequential(
            nn.Linear(400,3),

        )

        self.decommpress = nn.Sequential(
            nn.Linear(3,120),
            nn.ReLU(),
            nn.Linear(120,480),
            nn.ReLU(),
            nn.Linear(480,784),
        )
    
    def forward(self,x):
        output_conv = self.conv(x)
        hidden = self.compress(output_conv)
        out = self.decommpress(hidden)
        return (out,hidden)


class dataset():
    def __init__(self,amount_of_each=300,amount_numbers=10):
        transform = transforms.Compose([transforms.ToTensor()])

        self.train = torchvision.datasets.MNIST(".",train=True,transform=transform,download=True)
        self.amount_numbers = amount_numbers
        self.amount_of_each = amount_of_each
        self.numbers_ordered = self.order_num(self.train)
        
    
    def order_num(self,data):
        numbers_ordered = [[] for i in range(self.amount_numbers)]

        for num in tqdm(data,ascii=True,desc="prepare dataset"):
            if(len(numbers_ordered[num[1]]) <self.amount_of_each):
                numbers_ordered[num[1]].append(num[0]) 

        return numbers_ordered

    def sample_num(self,number):
        image = self.numbers_ordered[number][np.random.randint(len(self.numbers_ordered[number]))]
        return image.view(1,1,28,28)
    
    def sample_random_numbers(self,batch_size):
        numbers = np.random.choice(self.amount_numbers,batch_size)

        return torch.cat([self.sample_num(i) for i in numbers])
    
    def first_few(self,number,amount):
        return self.all_of_number(number)[:amount]

    def all_of_number(self,number):
        numbers = []
        for image in self.numbers_ordered[number]:
            numbers.append(image.view(1,1,28,28))
        return numbers



if __name__ == "__main__":
    #repeatability
    torch.manual_seed(0)
    np.random.seed(0)

    #obj
    net = network()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    dataset_MNIST = dataset()

    #hyperparam
    EPOCHS = 2000
    BATCH_SIZE = 150 
    plot_amount_numbers_each = 42

    #plotting loss
    x = np.arange(EPOCHS)
    y = []

    #train loop
    for _ in tqdm(range(EPOCHS),ascii=True,desc="train loop"):
        optimizer.zero_grad()
        
        input_d = dataset_MNIST.sample_random_numbers(BATCH_SIZE)
        output, hidden = net(input_d)
        loss = nn.MSELoss()(output,nn.Flatten()(input_d))
        loss.backward()
        optimizer.step()
        y.append(loss.detach().item())
    

    #plot loss 
    plt.title("Train loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(x,y,label="loss")
    plt.legend()
    plt.show()    


    #colors 
    cm = pylab.get_cmap('gist_rainbow')
    cgen = [cm(1.*i/dataset_MNIST.amount_numbers) for i in range(dataset_MNIST.amount_numbers)]
    
    #plot numbers 
    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    
    
    plt.title("hidden state")

    for num in tqdm(range(dataset_MNIST.amount_numbers),ascii=True,desc="plotting numbers",unit="num"):
        color = cgen[num]
        _,hidden = net(torch.cat(dataset_MNIST.first_few(num,plot_amount_numbers_each)))
        x,y,z = list(zip(*hidden.detach().numpy()))

        axis.scatter(x,y,z,color=color,label=str(num))

    axis.legend()
    plt.show()
           

