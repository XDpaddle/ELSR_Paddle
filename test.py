import argparse
import matplotlib.pyplot as plt
import cv2

import paddle
from model import ELSR
from preprocessing import psnr, prepare_img

def test_image(model,  img, upscale_factor):
    img = cv2.resize(img, (img.shape[1]//upscale_factor*upscale_factor, img.shape[0]//upscale_factor*upscale_factor), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.resize(img, (img.shape[1]//upscale_factor, img.shape[0]//upscale_factor), interpolation=cv2.INTER_CUBIC)
    bicubic_upscaled = cv2.resize(lr_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    lr_img = prepare_img(lr_img)
    bicubic_upscaled = prepare_img(bicubic_upscaled)
    img = prepare_img(img)

    with paddle.no_grad():
        sr_img = paddle.clip(model(lr_img),0,1)

    return sr_img, bicubic_upscaled, img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="best_X4.pdparams")
    parser.add_argument('--input', type=str, default="test/sonic.jpg")
    parser.add_argument('--scale', type=int,default=4)
    args = parser.parse_args()



    model = ELSR(upscale_factor=args.scale)
    #paddle.save(model.state_dict(),'test.pdparams')


    state_dict = paddle.load(args.weights)
    model.set_state_dict(state_dict=state_dict)
    model.eval()

    image = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)

    sr_img, bicubic_upscaled, image = test_image(model,  image, upscale_factor=args.scale)

    psnr_value = psnr(sr_img, image).numpy()
    bicubic_psnr = psnr(bicubic_upscaled, image).numpy()
    print(f"PSNR of Bicubic upscaled: {bicubic_psnr} dB")
    print(f"PSNR of Super-resoluted image: {psnr_value} dB")

    #Save images
    
    out = sr_img.numpy().squeeze(0).transpose(1, 2, 0)
    bicubic_out = bicubic_upscaled.numpy().squeeze(0).transpose(1, 2, 0)

    plt.imsave("out/output.png", out)
    plt.imsave("out/bicubic.png", bicubic_out)
