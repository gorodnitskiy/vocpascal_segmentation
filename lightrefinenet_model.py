import torch
import segmentation_models_pytorch as smp


class ConvNormAct(torch.nn.Module):
    def __init__(self,
                 *args,
                 activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm2d,
                 **kwargs):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(*args, **kwargs),
            normalization(args[1]),
            activation()
        )

    def forward(self, input):
        return self.module(input)


class Residual(torch.nn.Module):
    def __init__(self, block,
                 bypass=None):
        super().__init__()
        self.block = block
        self.bypass = bypass

    def forward(self, input):
        if self.bypass is None:
            return self.block(input) + input
        else:
            return self.block(input) + self.bypass(input)


class RCU(torch.nn.Module):
    def __init__(self, *args,
                 activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm2d,
                 **kwargs):
        super().__init__()

        self.module = Residual(
            torch.nn.Sequential([
                ConvNormAct(*args, activation=activation,
                            normalization=normalization, **kwargs),
                ConvNormAct(*args, activation=activation,
                            normalization=normalization, **kwargs)]))

    def forward(self, x):
        return self.module(x)


class CRP(torch.nn.Module):
    def __init__(self, *args,
                 n=2,
                 activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm2d,
                 **kwargs):
        """
        оптимизатор может не понимать, что блоки нужно оптимизировать
        не будет переключать eval как бы, то есть на валидации будет идти
        обучение. На одну и ту же картинку будут разные предикты в разных
        батчах. Есть сделать self.block = torch.nn.ModuleList(self.block), то
        все норм будет
        """
        super().__init__()

        self.blocks = []

        for index in range(n):
            self.blocks.append(torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvNormAct(*args, activation=activation,
                            normalization=normalization, **kwargs)))

        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x):
        res = x.clone()

        for block in self.blocks:
            x = block(x)
            res = res + x

        return res


class LightRefineNet(torch.nn.Module):
    def __init__(self, activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm2d,
                 n_classes=20):
        super().__init__()
        self.backbone = smp.encoders.get_encoder(
            'efficientnet-b0', weights='imagenet')

        self.decoder = []
        for index in reversed(range(1, len(self.backbone.out_channels) - 1)):
            in_channels = self.backbone.out_channels[index + 1]
            out_channels = self.backbone.out_channels[index]
            self.decoder.append(
                torch.nn.Sequential(
                    CRP(in_channels,
                        in_channels,
                        1,
                        n=4,
                        activation=activation,
                        normalization=normalization),
                    torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    1),
                    torch.nn.UpsamplingBilinear2d(scale_factor=2)))

        self.decoder = torch.nn.ModuleList(self.decoder)[::-1]

        self.final = torch.nn.Sequential(
            CRP(self.backbone.out_channels[1], self.backbone.out_channels[1], 1),
            torch.nn.Conv2d(self.backbone.out_channels[1], n_classes, 1),
            torch.nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x):
        res = self.backbone(x)
        for index in reversed(range(1, len(self.backbone.out_channels) - 1)):
            res[index] = res[index] + self.decoder[index - 1](res[index + 1])

        return self.final(res[1])
