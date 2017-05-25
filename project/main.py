import tensorflow as tf
import utils


def main():
    data = utils.load_tiny_imagenet('./data')
    print(data['X_train'].shape)
    print(data['y_train'].shape)
    print(data['X_val'].shape)
    print(data['y_val'].shape)
    print(data['X_test'].shape)
    print(data['y_test'].shape)
    print(data['class_names'].shape)
    print(data['mean_image'].shape)

if __name__ == "__main__":
    main()
