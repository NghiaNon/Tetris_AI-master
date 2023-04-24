import torch.nn as nn

#cấu trúc của mạng thần kinh sẽ được sử dụng để tính gần đúng các giá trị Q của các cặp trạng thái-hành động.
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

#Mạng có 3 lớp kết nối đầy đủ :
        # Lớp đầu tiên ở trạng thái được biểu thị bằng một tensor 4 chiều(số dòng đã xóa, số lỗ, độ gập ghềnh và chiều cao)
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

#khởi tạo trọng số của tất cả các lớp tuyến tính trong mạng bằng cách sử dụng khởi tạo Xavier và đặt độ lệch thành 0
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

#Phương thức chuyển tiếp của lớp DeepQNetwork đang triển khai chuyển tiếp chuyển tiếp của mạng thần kinh.
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
