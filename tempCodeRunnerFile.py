for i in vector:
    np.random.shuffle(xrange)

    canny = cv2.Canny(image,i,3*i)
    canny = cv2.dilate(canny,kernel,iterations=1)

    for j in xrange:
        np.random.shuffle(yrange)
        for k in yrange:
            if canny[j,k] == 255:
                x = j + randrange(-jitter, jitter)
                y = k + randrange(-jitter, jitter)
                pixel = img[j, k].tolist()
                bola = int(raio + (i)/190*randrange(2,5))
                cv2.circle(points, (y, x), bola, pixel, -1, cv2.LINE_AA)