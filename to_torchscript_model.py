import os
import sys
import torch

from craft import CRAFT
import torch.backends.cudnn as cudnn

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_model(bpth_file_path):
    num_classes = len(classes)
    cuda = torch.cuda.is_available()
    model = net_model.load_model(bpth_file_path, num_classes, cuda)
    model.eval()
    return model


def main(pth_file_path):
    cuda = True
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + pth_file_path + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(pth_file_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(pth_file_path, map_location='cpu')))

    if cuda:
        net = net.cuda()
        cudnn.benchmark = False

    net.eval()

    script_module = torch.jit.script(net)

    file_path_without_ext = os.path.splitext(pth_file_path)[0]
    output_file_path = file_path_without_ext + ".pt"
    script_module.save(output_file_path)
    print("TorchScript model created:", output_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid arguments!")
        print("Usage: python3 " + sys.argv[0] + " <pth_file_path>")
    else:
        main(sys.argv[1])
