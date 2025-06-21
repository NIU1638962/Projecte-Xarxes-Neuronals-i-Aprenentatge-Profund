# -*- coding: utf-8 -*- noqa
"""
Created on Wed Apr 30 05:25:22 2025

@author: Joel Tapia Salvador
"""
from abc import ABCMeta, abstractclassmethod

import environment
import utils


class ConditionalDropout2d(environment.torch.nn.Module):
    def __init__(self, p=0.5, active=True):
        super().__init__()
        self.p = p
        self.active = active
        self.dropout = environment.torch.nn.Dropout2d(p)

    def forward(self, x):
        if self.training and self.active:
            return self.dropout(x)
        return x


class __MultiLabelClassifier(environment.torch.nn.Module, metaclass=ABCMeta):

    def __init__(
            self,
            input_dimensions,
            number_classes=[30],
            threshold_method='roc_closest',
    ):

        super().__init__()

        self.number_classes = number_classes

        self.sigmoid = environment.torch.nn.Sigmoid()

        self.sigmoid.should_initialize = False

        self.threshold_method = threshold_method

        self.thresholds = environment.torch.nn.ParameterList(
            [
                environment.torch.nn.Parameter(
                    environment.torch.full((num_classes,), 0.5),
                    requires_grad=False,
                ) for num_classes in self.number_classes]
        )

        # self.register_buffer(
        #     'thresholds',
        #     [
        #         environment.torch.full(
        #             (num_classes,),
        #             0.5,
        #         ) for num_classes in self.number_classes
        #     ],
        # )

    @abstractclassmethod
    def forward(x):
        pass

    @environment.torch.no_grad()
    def results(self, list_logits):
        results = []
        for logits, threshold in zip(list_logits, self.thresholds):
            results.append(
                (
                    self.sigmoid(logits) >= threshold
                ).to(environment.torch.float32)
            )
        return results

    @environment.torch.no_grad()
    def update_threshold(self, list_logits, list_labels):
        if not self.training:
            return

        utils.print_message('Updating thresholds...')

        for (
            logits,
            labels,
            thresholds,
            number_classes,
        ) in zip(
            list_logits,
            list_labels,
            self.thresholds,
            self.number_classes,
        ):

            probabilities = self.sigmoid(logits)

            for c in range(number_classes):
                if self.threshold_method == 'pr':
                    thresholds.data[c] = self._update_threshold_pr(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()
                elif self.threshold_method == 'roc_youden':
                    thresholds.data[c] = self._update_threshold_roc_youden(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()
                elif self.threshold_method == 'roc_closest':
                    thresholds.data[c] = self._update_threshold_roc_closest(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()

        utils.print_message('Thresholds updated.')

    @environment.torch.no_grad()
    def _update_threshold_roc_closest(self, probabilities, labels):
        fpr, tpr, thresholds = environment.sklearn.metrics.roc_curve(
            labels.cpu(),
            probabilities.cpu(),
        )

        return environment.torch.tensor(
            thresholds[
                environment.numpy.argmin(
                    environment.numpy.sqrt((1 - tpr) ** 2 + fpr ** 2)
                )
            ],
            dtype=environment.torch.float32,
            device=environment.TORCH_DEVICE,
        )


class Baseline(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        hidden_dim=128,
        classifier_hidden_layers=[],
        number_classes=[30],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        self.feature_extractor = environment.torch.nn.Conv1d(
            input_dimensions[0],
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        for module in self.feature_extractor.modules():
            module.should_initialize = True

        self.encoder = environment.torch.nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
        )

        for module in self.encoder.modules():
            module.should_initialize = True

        layers = []
        input_dimension = hidden_dim

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes[0],
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"
        z = self.feature_extractor(x)
        assert not environment.torch.isnan(z).any(), "NaN after conv"

        z = environment.torch.nn.functional.relu(z)
        z = z.transpose(2, 1)

        features = self.encoder(z)[0][:, -1, :]
        assert not environment.torch.isnan(features).any(), "NaN after GRU"

        features = environment.torch.nn.functional.relu(features)
        logits = self.classifier(features)
        assert not environment.torch.isnan(logits).any(), "NaN in logits"

        return [logits]


class FullCNN(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        hidden_dim=64,
        classifier_hidden_layers=[],
        number_classes=[30],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        self.conv_layers = environment.torch.nn.Sequential(
            environment.torch.nn.Conv2d(
                1,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            environment.torch.nn.ReLU(),
            environment.torch.nn.BatchNorm2d(hidden_dim),
            environment.torch.nn.Dropout2d(p=dropout_rate),
            environment.torch.nn.MaxPool2d(kernel_size=2, stride=2),

            environment.torch.nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            environment.torch.nn.ReLU(),
            environment.torch.nn.BatchNorm2d(hidden_dim * 2),
            environment.torch.nn.Dropout2d(p=dropout_rate),
            environment.torch.nn.MaxPool2d(kernel_size=2, stride=2),

            environment.torch.nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            environment.torch.nn.ReLU(),
            environment.torch.nn.AdaptiveAvgPool2d((1, 1)),
            environment.torch.nn.Flatten(),
        )

        for module in self.conv_layers.modules():
            module.should_initialize = True

        layers = []
        input_dimension = self.conv_layers[-4].out_channels

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes[0],
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"
        x = x.unsqueeze(1)
        z = self.conv_layers(x)
        assert not environment.torch.isnan(z).any(), "NaN in conv layers"
        logits = self.classifier(z)
        return [logits]


class ResNet18(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        self.should_initialize = True

        resnet = environment.torchvision.models.resnet18(
            weights=environment.torchvision.models.ResNet18_Weights.DEFAULT,
        )

        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = environment.torch.nn.Sequential(
            *list(resnet.children())[:-1],
            environment.torch.nn.Flatten(),
        )

        for module in self.feature_extractor.modules():
            module.should_initialize = False

        self.last_layer_output_dim = environment.deepcopy(
            resnet.fc.in_features
        )

        layers = []
        input_dimension = self.last_layer_output_dim

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes[0],
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        z = self.feature_extractor(x)
        logits = self.classifier(z)
        return [logits]


class ResNet18FineTuneLayer4(ResNet18):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(
            input_dimensions,
            number_classes,
            classifier_hidden_layers,
            threshold_method,
            dropout_rate,
        )

        for param in self.feature_extractor[-3].parameters():
            param.requires_grad = True

        original_layer4 = self.feature_extractor[-3]
        modified_blocks = []

        for block in original_layer4:
            modified_block = self._insert_dropout_in_block(block, dropout_rate)
            modified_blocks.append(modified_block)

        self.feature_extractor[-3] = environment.torch.nn.Sequential(
            *modified_blocks
        )

        for module in self.feature_extractor.modules():
            module.should_initialize = False

    def _insert_dropout_in_block(self, block, dropout_rate):
        """
        Inserts Dropout2d before each convolution in a BasicBlock.
        Returns a modified BasicBlock.
        """

        # Create a copy to avoid in-place changes
        new_block = environment.deepcopy(block)

        # Replace conv1 and conv2 with dropout + conv + rest of the block
        new_block.conv1 = environment.torch.nn.Sequential(
            environment.torch.nn.Dropout2d(p=dropout_rate),
            new_block.conv1,
        )

        new_block.conv1[0].should_initialize = False

        new_block.conv2 = environment.torch.nn.Sequential(
            environment.torch.nn.Dropout2d(p=dropout_rate),
            new_block.conv2,
        )

        new_block.conv2[0].should_initialize = False

        return new_block


class ResNet18FineTuneLayer4LastBlock(ResNet18):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(
            input_dimensions,
            number_classes,
            classifier_hidden_layers,
            threshold_method,
            dropout_rate,
        )

        for param in self.feature_extractor[-3][-1].parameters():
            param.requires_grad = True

        original_layer4_last_block = self.feature_extractor[-3][-1]

        modified_block = self._insert_dropout_in_block(
            original_layer4_last_block,
            dropout_rate,
        )

        self.feature_extractor[-3][-1] = environment.torch.nn.Sequential(
            modified_block
        )

    def _insert_dropout_in_block(self, block, dropout_rate):
        """
        Inserts Dropout2d before each convolution in a BasicBlock.
        Returns a modified BasicBlock.
        """

        # Create a copy to avoid in-place changes
        new_block = environment.deepcopy(block)

        # Replace conv1 and conv2 with dropout + conv + rest of the block
        new_block.conv1 = environment.torch.nn.Sequential(
            environment.torch.nn.Dropout2d(p=dropout_rate),
            new_block.conv1,
        )

        new_block.conv1[0].should_initialize = False

        new_block.conv2 = environment.torch.nn.Sequential(
            environment.torch.nn.Dropout2d(p=dropout_rate),
            new_block.conv2,
        )

        new_block.conv2[0].should_initialize = False

        return new_block


class ResNet18MultiHead(ResNet18FineTuneLayer4LastBlock):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(
            input_dimensions,
            number_classes,
            classifier_hidden_layers,
            threshold_method,
            dropout_rate,
        )

        self.classifier = environment.torch.nn.ModuleList()
        self.classifier.should_initialize = True

        for num_classes in self.number_classes:
            layers = []
            input_dimension = self.last_layer_output_dim

            for hidden_dimension in classifier_hidden_layers:
                layers.append(environment.torch.nn.Dropout(p=dropout_rate))
                layers.append(environment.torch.nn.Linear(
                    input_dimension,
                    hidden_dimension,
                ))
                layers.append(environment.torch.nn.ReLU())
                input_dimension = hidden_dimension

            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                num_classes,
            ))

            head = environment.torch.nn.Sequential(*layers)

            for module in head.modules():
                module.should_initialize = True

            self.classifier.append(head)

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        z = self.feature_extractor(x)
        logits = [head(z) for head in self.classifier]
        return logits


class ResNet50(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        resnet = environment.torchvision.models.resnet50(
            weights=environment.torchvision.models.ResNet50_Weights.DEFAULT,
        )

        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = environment.torch.nn.Sequential(
            *list(resnet.children())[:-1],
            environment.torch.nn.Flatten(),
        )

        for module in self.feature_extractor.modules():
            module.should_initialize = False

        layers = []
        input_dimension = resnet.fc.in_features

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes[0],
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        z = self.feature_extractor(x)
        logits = self.classifier(z)
        return [logits]


class ResNet50FineTuneLayer4(ResNet50):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
        train_feature_estractor=True,
    ):
        super().__init__(
            input_dimensions,
            number_classes,
            classifier_hidden_layers,
            threshold_method,
            dropout_rate,
        )

        for param in self.feature_extractor[-3].parameters():
            param.requires_grad = True

        modules = list(self.feature_extractor.children())
        modules.insert(-3, environment.torch.nn.Dropout2d(p=dropout_rate))
        self.feature_extractor = environment.torch.nn.Sequential(*modules)


class ResNet101(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        resnet = environment.torchvision.models.resnet101(
            weights=environment.torchvision.models.ResNet101_Weights.DEFAULT,
        )

        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = environment.torch.nn.Sequential(
            *list(resnet.children())[:-1],
            environment.torch.nn.Flatten(),
        )

        for module in self.feature_extractor.modules():
            module.should_initialize = False

        layers = []
        input_dimension = resnet.fc.in_features

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes[0],
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        z = self.feature_extractor(x)
        logits = self.classifier(z)
        return [logits]


class ResNet101FineTuneLayer4(ResNet101):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
    ):
        super().__init__(
            input_dimensions,
            number_classes,
            classifier_hidden_layers,
            threshold_method,
            dropout_rate,
        )

        for param in self.feature_extractor[-3].parameters():
            param.requires_grad = True

        modules = list(self.feature_extractor.children())
        modules.insert(-3, environment.torch.nn.Dropout2d(p=dropout_rate))
        self.feature_extractor = environment.torch.nn.Sequential(*modules)


class PANNsCNN14(__MultiLabelClassifier):
    def __init__(
        self,
        input_dimensions,
        number_classes=[30],
        classifier_hidden_layers=[],
        threshold_method='roc_closest',
        dropout_rate=0.5,
        pretrained_path='Cnn14.pth',
    ):
        super().__init__(input_dimensions, number_classes, threshold_method)

        # === Definición del extractor ===
        self.feature_extractor = environment.torch.nn.Sequential(
            self._conv_block(1, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            self._conv_block(512, 1024),
        )

        if pretrained_path is not None:
            state = environment.torch.load(pretrained_path, map_location='cpu')
            if 'model' in state:
                state = state['model']
            self.feature_extractor.load_state_dict(
                {k.replace('cnn14.', '').replace('module.', ''): v
                 for k, v in state.items() if 'fc' not in k},
                strict=False
            )
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.pool = environment.torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = environment.torch.nn.Dropout(0.5)

        layers = []
        input_dimension = None  # how to get from last layer output of feature extractor

        for hidden_dimension in classifier_hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes,
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels):
        return environment.torch.nn.Sequential(
            environment.torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1),
            environment.torch.nn.BatchNorm2d(out_channels),
            environment.torch.nn.ReLU(),
            environment.torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1),
            environment.torch.nn.BatchNorm2d(out_channels),
            environment.torch.nn.ReLU(),
            environment.torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        assert not environment.torch.isnan(x).any(), "NaN in input"
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return [logits]


AVAILABLE_MODELS = {
    'baseline': Baseline,
    'full_cnn': FullCNN,
    'res_net_18': ResNet18,
    'res_net_18_fine_tune_layer_4': ResNet18FineTuneLayer4,
    'res_net_18_fine_tune_layer_4_last_block': ResNet18FineTuneLayer4LastBlock,
    'res_net_18_multi_head': ResNet18MultiHead,
    'res_net_50': ResNet50,
    'res_net_50_fine_tune_layer_4': ResNet50FineTuneLayer4,
    'res_net_101': ResNet101,
    'res_net_101_fine_tune_layer_4': ResNet101FineTuneLayer4,
    'panns_cnn_14': PANNsCNN14,
}
