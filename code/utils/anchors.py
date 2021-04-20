import numpy as np


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """

    :param anchor_base:       size:（9*4）,特征图每个点相对于(0,0)的尺寸(x1,y1,x2,y2)to(0,0)
    :param feat_stride:       表示原图是特征图的feat_stride倍
    :param height:            特征图的高
    :param width:             特征图的宽
    :return:                  anchors在原图中对应坐标
    """

    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride) #(1,width)
    shift_y = np.arange(0, height * feat_stride, feat_stride)#(1,height)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)    #(height,width),(height,width)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),  #(height*width,4)
                      shift_x.ravel(),shift_y.ravel(),), axis=1)
    #shift存储的是特征图上所有点的中心坐标shift[0]，shift[2]都表示所有的x坐标，shift[1],shift[3]都表示所有的y坐标
    #之所以写两次是因为每个点的anchor有四个值(左上x,y偏移,右下x,y偏移）
    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0] #特征图的点数（height*width)
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4)) #利用broadcast 输出(K,A,4)
    # 所有的anchors
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride = 38,38,16
    anchors_all = _enumerate_shifted_anchor(nine_anchors,feat_stride,height,width)
    print(np.shape(anchors_all))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108,109,110,111,112,113,114,115,116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    
    plt.show()
