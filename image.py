# show images in a window from fer2013.csv
from utils import getData
import matplotlib.pyplot as plt
import numpy as np

# label mapping
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# show one image for each label
def main():
    X,Y,_,_ = getData()
    _ = 0
    while _ < 7:
        for i in range(7):
            x,y = X[Y==i], Y[Y==i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48,48), cmap='gray')
            plt.title(labels[i])
            plt.axis('off')
            plt.show()
            _ += 1

if __name__ == '__main__':
    main()