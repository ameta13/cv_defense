import argparse

import timm
from torch.utils.data import DataLoader

from cv_defense.datasets import CustomDataset
from cv_defense.helpers.training import train_model, evaluate


def get_model(name_model: str, num_classes: int):
    return timm.create_model(name_model, pretrained=True, num_classes=num_classes)


parser = argparse.ArgumentParser(
    prog='Dataset defender',
    description='Detector adversarial examples in dataset of images',
)
# Dataset params
parser.add_argument('--dataset-root-path', type=str, required=True)
parser.add_argument('--num-classes', type=int, required=True)
parser.add_argument('--path_to_train', type=str, default='train.csv')
parser.add_argument('--column_path_to_image', type=str, default='path')
parser.add_argument('--column_label', type=str, default='label')
# Output params
parser.add_argument('--output-path', type=str, default='./adversarial_examples_indexes.csv')
# Training params
parser.add_argument(
    '--base-model',
    type=str,
    default='resnet18',
    help='Model for training on the provided dataset',
    choices=timm.list_models(pretrained=True),
)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size of images')
parser.add_argument('--device', type=str, default='cuda')


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = CustomDataset(
        args.dataset_root_path,
        train=True,
        path_to_train=args.path_to_train,
        column_path=args.column_path_to_image,
        column_label=args.column_label,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = get_model(args.base_model, args.num_classes).to(args.device)
    train_model(model, dataloader, num_epochs=self.num_epochs, device=self.device)
    defense = ActivationClusteringDefense(model, args.num_classes, device=args.device)

    clean_model = get_model(args.base_model, args.num_classes).to(args.device)
    example_idxs = defense.analyze_by_exclusion(dataloader, clean_model)

    print(f'Find {len(example_idxs)} adversarial images')
    if example_idxs:
        print(f'Saving adversarial images to the {args.output_path}')
        with open(args.output_path, 'w') as f:
            f.write('\n'.join(example_idxs))
