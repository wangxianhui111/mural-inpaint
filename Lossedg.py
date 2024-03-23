def get_edge(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ratio = 3
    lowThreshold = 80
    kernel_size = 3
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio,
                               apertureSize=kernel_size)  # 边缘检测
    dst = detected_edges
    dst1 = torch.from_numpy(dst.astype(np.float32) / 255.0).contiguous().cuda()
    dst2 = dst1.unsqueeze(0)
    return dst2


def getcount(img):
    count2 = 0
    for i in range(0, 256):
        for j in range(0, 256):
            if img[i, j] == 1:
                count2 = count2 + 1
    return count2


def getedgeloss(img2, sketch):
    # loss1=0
    loss2 = 0
    skecount = 0
    for i in range(0, 256):
        for j in range(0, 256):
            if sketch[i, j] == 1:
                skecount = skecount + 1
                # loss1=((img1[i,j]-sketch[i,j]).abs()+loss1).cuda()
                loss2 = ((img2[i, j] - sketch[i, j]).abs() + loss2).cuda()
    return skecount, loss2


def getlossedge(img, mask, sketch):
    img_edg = get_edge(fromtourch(img)).cuda()
    img_edg32 = torch.unsqueeze(torch.cat((img_edg, img_edg, img_edg), 0), 0).cuda()
    img_edg22 = img_edg32[0, 0, :, :].cuda()
    onlysketch = mask * (sketch)
    onlysketch22 = onlysketch[0, 0, :, :]
    count2 = getcount(onlysketch22)
    img = onlysketch22 * img_edg22.cuda()
    loss_edge = ((img - sketch).abs().mean() / count2) * 65536
    return loss_edge


def getlossedge2(img1, img2, mask, sketch):
    onlysketch = mask * (sketch)
    onlysketch22 = onlysketch[0, 0, :, :]
    count2 = getcount(onlysketch22)

    img1_edg = get_edge(fromtourch(img1)).cuda()
    img1_edg32 = torch.unsqueeze(torch.cat((img1_edg, img1_edg, img1_edg), 0), 0).cuda()
    img1_edg22 = img1_edg32[0, 0, :, :].cuda()
    img1 = onlysketch22 * img1_edg22.cuda()
    loss1_edge = ((img1 - sketch).abs().mean() / count2) * 65536

    img2_edg = get_edge(fromtourch(img2)).cuda()
    img2_edg32 = torch.unsqueeze(torch.cat((img2_edg, img2_edg, img2_edg), 0), 0).cuda()
    img2_edg22 = img2_edg32[0, 0, :, :].cuda()
    img2 = onlysketch22 * img2_edg22.cuda()
    loss2_edge = ((img2 - sketch).abs().mean() / count2) * 65536
    return loss1_edge, loss2_edge


def getlossedge3(img2, mask, sketch):
    onlysketch = mask * (sketch)
    onlysketch22 = onlysketch[0, 0, :, :]
    # img1_edg = get_edge(fromtourch(img1)).cuda()
    # img1_edg32 = torch.unsqueeze(torch.cat((img1_edg, img1_edg, img1_edg), 0), 0).cuda()
    # img1_edg22 = img1_edg32[0, 0, :, :].cuda()
    # img1 = onlysketch22 * img1_edg22.cuda()
    # img11=onlysketch*img1_edg32
    img2_edg = get_edge(fromtourch(img2)).cuda()
    img2_edg32 = torch.unsqueeze(torch.cat((img2_edg, img2_edg, img2_edg), 0), 0).cuda()
    img2_edg22 = img2_edg32[0, 0, :, :].cuda()
    img2 = onlysketch22 * img2_edg22.cuda()
    img22 = onlysketch * img2_edg32
    count, loss2 = getedgeloss(img2, onlysketch22)
    if count != 0:
        # loss1_edge = loss1 / count
        loss2_edge = loss2 / count
    else:
        # loss1_edge = 0
        loss2_edge = 0
    return loss2_edge, img22, onlysketch

