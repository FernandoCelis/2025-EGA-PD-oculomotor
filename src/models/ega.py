import torch
import torch.nn as nn
import libs.spdnetwork.nn as nn_spd


class EGA(nn.Module):
    def __init__(self, dtype, device, in_channels=3):
        super(EGA, self).__init__()

        self.TridiConv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.covariancePooling = nn_spd.CovPool()
        self.bire_1 = nn.Sequential(
            nn_spd.BiMap(1, 1, 128, 64, dtype=dtype, device=device),
            nn_spd.ReEig()
        )
        self.ega = nn_spd.EGA(dim=64, output_dim=32, dtype=dtype, device=device)
        self.log_eig = nn_spd.LogEig()
        self.linear = torch.nn.Linear(32 * (32 + 1) // 2, 1, dtype=torch.double)
        self.indices = torch.triu_indices(32, 32)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        x = self.TridiConv1(inputs)
        x = x.view(x.size(0), x.size(1), -1)
        x_spd = self.covariancePooling(x)
        x_spd = self.bire_1(x_spd)
        x_ega = self.ega(x_spd)
        output = x_ega[:, :, self.indices[0], self.indices[1]]
        output = output.view(output.shape[0], -1)
        pred = self.linear(output)
        return pred


class CrossATT_64Bire_32AttLEFT(nn.Module):
    def __init__(self, dtype, device, in_channels=3):
        super(CrossATT_64Bire_32AttLEFT, self).__init__()

        self.TridiConv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.TridiConv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.covariancePooling = nn_spd.CovPool()
        self.birel_1 = nn.Sequential(nn_spd.BiMap(1, 1, 128, 64, dtype=dtype, device=device), nn_spd.ReEig())
        self.birer_1 = nn.Sequential(nn_spd.BiMap(1, 1, 128, 64, dtype=dtype, device=device), nn_spd.ReEig())

        self.ega = nn_spd.EGA(dim=64, output_dim=32, dtype=dtype, device=device)

        self.linear = torch.nn.Linear(32 * (32 + 1) // 2, 1, dtype=torch.double)
        self.indices = torch.triu_indices(32, 32)

    def forward(self, inputs):
        inputs_left, inputs_right = inputs
        inputs_left = inputs_left.permute(0, 2, 1, 3, 4)
        inputs_right = inputs_right.permute(0, 2, 1, 3, 4)

        x_left = self.TridiConv1(inputs_left)
        x_right = self.TridiConv2(inputs_right)

        x_left = x_left.view(x_left.size(0), x_left.size(1), -1)
        x_left = self.covariancePooling(x_left)

        x_right = x_right.view(x_right.size(0), x_right.size(1), -1)
        x_right = self.covariancePooling(x_right)

        x_left = self.birel_1(x_left)
        x_right = self.birer_1(x_right)

        x_ega = self.ega(x_left, x_right)

        x = x_ega[:, :, self.indices[0], self.indices[1]]
        x = x.view(x.shape[0], -1)
        return self.linear(x)
 