import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=7):
        super(InitialBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size)
        )

    def forward(self, x):
        output = self.layers(x)
        if output.shape != x.shape:
            return output + torch.cat((x, x), axis=1)
        return x + output


class RefineBlock(nn.Module):
    def __init__(self, input_size, output_size, upscale_factor=2):
        super(RefineBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_size),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        return self.layers(x)


class FinalBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super(FinalBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)


class RUNet(nn.Module):
    def __init__(self):
        super(RUNet, self).__init__()
        self.block1 = InitialBlock(3, 64)
        
        self.block2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128)
        )

        self.block3 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256)
        )

        self.block4 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512)
        )

        self.block5 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.representation_transform = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.refine4 = RefineBlock(1024, 512)
        self.refine3 = RefineBlock(512 + 512//4, 384)
        self.refine2 = RefineBlock(256 + 384//4, 256)
        self.refine1 = RefineBlock(128 + 256//4, 96)

        self.final = FinalBlock(64 + 96//4, 99, 3)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.block1(x)

        x2 = self.block2(self.max_pool(x1))
        
        x3 = self.block3(self.max_pool(x2))

        x4 = self.block4(self.max_pool(x3))

        x5 = self.block5(self.max_pool(x4))

        output5 = self.representation_transform(x5)

        input5 = torch.cat([x5, output5], dim=1)
        output4 = self.refine4(input5)

        input4 = torch.cat([x4, output4], dim=1)
        output3 = self.refine3(input4)

        input3 = torch.cat([x3, output3], dim=1)
        output2 = self.refine2(input3)

        input2 = torch.cat([x2, output2], dim=1)
        output1 = self.refine1(input2)

        input1 = torch.cat([x1, output1], dim=1)
        output = self.final(input1)

        return output
