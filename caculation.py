def wocao(channel,ks):
    return 8*(ks**2)*(channel**2)+4*channel + 26*channel*(ks**2)

def lstu_weight(channel,ks):
    return 4*(channel**2)*(ks**2) + 13*channel*(ks**2)+3*channel

def flops(channel, width):
    return 117 * (channel**2) * (width**2) - 3.25*(width**2)*channel + 4.5 * 13 *channel*(width**2) + 2*channel*(width**2)

def lstu_flops(channel, width):
    return 45 * (channel**2) * (width**2) +1.75*(width**2)*channel + 4.5 * 13 *channel*(width**2)
if __name__ == "__main__":
    channel = 64
    ks    = 3
    total = 0
    for i in range(5):
        wori = 2**(i-1)
        total += wocao(channel*wori,ks )

    print("stu %f"%total)

    # channel = 16
    # input_image= 128
    # total = 0
    # for i in range(5):
    #     wori = 2**(i-1)
    #     input_image = input_image/wori
    #     total += lstu_flops(channel*wori, input_image)

    # print(total)