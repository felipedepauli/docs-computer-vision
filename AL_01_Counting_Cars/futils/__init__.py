def centroide (x, y, w, h):
    """
    :param x: x coordinate of the upper left corner of the rectangle
    :param y: y coordinate of the upper left corner of the rectangle
    :param w: width of the rectangle
    :param h: height of the rectangle
    
    return: tuple (x, y) of the center of the rectangle
    """
    x0  = w // 2
    y0 = h // 2
    cx = x + x0
    cy = y + y0
    
    return cx, cy 
