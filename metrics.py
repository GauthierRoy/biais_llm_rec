def calc_iou(x, y):
    x = set(x)
    y = set(y)
    return len(x & y) / len(x | y)


def calc_serp_ms(x, y):
    temp = 0
    if len(y) == 0:
        return 0
    for i, item_x in enumerate(x):
        for j, item_y in enumerate(y):
            if item_x == item_y:
                temp = temp + len(x) - i + 1
    return temp * 0.5 / ((len(y) + 1) * len(y))


def calc_prag(x, y):
    temp = 0
    sum = 0
    if len(y) == 0 or len(x) == 0:
        return 0
    if len(x) == 1:
        if x == y:
            return 1
        else:
            return 0
    for i, item_x1 in enumerate(x):
        for j, item_x2 in enumerate(x):
            if i >= j:
                continue
            id1 = -1
            id2 = -1
            for k, item_y in enumerate(y):
                if item_y == item_x1:
                    id1 = k
                if item_y == item_x2:
                    id2 = k
            sum = sum + 1
            if id1 == -1:
                continue
            if id2 == -1:
                temp = temp + 1
            if id1 < id2:
                temp = temp + 1
    return temp / sum
