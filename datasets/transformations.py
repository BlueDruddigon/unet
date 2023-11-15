from torchvision import transforms as T

__all__ = ['get_default_transformations']


def get_default_transformations(img_size):
    return {
      'train': T.Compose([
          T.Resize([img_size, img_size], antialias=True),
          T.RandomChoice([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
          ]),
          T.RandomRotation((-0.2, 0.2)),
      ]),
      'valid': T.Compose([
          T.Resize([img_size, img_size], antialias=True),
      ]),
      'test': T.Compose([
          T.Resize([img_size, img_size], antialias=True),
      ])
    }
