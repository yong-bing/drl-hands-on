from torch.utils.tensorboard import SummaryWriter
import math

if __name__ == '__main__':
    writer = SummaryWriter('runs/exp1')
    funcs = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan}
    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
    writer.close()