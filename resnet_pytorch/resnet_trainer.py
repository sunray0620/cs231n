import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def image_show(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.interactive(False)
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show(block=True)


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    image_show(torchvision.utils.make_grid(images))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == "__main__":
    main()