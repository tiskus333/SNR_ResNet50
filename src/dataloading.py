from torchvision import transforms

def load_data(train=0.8,valid=0.1,test=0.1,batch_size=64):

    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256),#, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            #transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])  # Imagenet standard
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    # Datasets from folders
    train_dataset = datasets.ImageFolder(root='Images', transform=image_transforms['train'])
    valid_dataset = datasets.ImageFolder(root='Images', transform=image_transforms['valid'])

    total_len = len(train_dataset)
    train_size = train
    valid_size = valid
    test_size = test
    indices = list(range(total_len))
    np.random.shuffle(indices)

    # Dataloader iterators, make sure to shuffle
    train_data = Subset(train_dataset,indices=indices[:int(np.floor(train_size*total_len))])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = Subset(valid_dataset ,indices=indices[int(np.floor((train_size-1)*total_len)):int(np.floor(-valid_size*total_len))])
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data = Subset(valid_dataset ,indices=indices[int(np.floor((train_size+valid_size-1)*total_len)):])
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader
