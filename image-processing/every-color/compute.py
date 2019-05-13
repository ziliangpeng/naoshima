
def c(m, n):
    ans = 1
    for i in range(n):
        ans *= m - i
        ans /= i + 1
    return int(ans)


def colors_of_brightness(br):
    if br < 0 or br > 255 * 3:
        return []

    ret = []
    for r in range(256):
        if r > br:
            continue
        if r + 255 + 255 < br:
            continue
        for g in range(256):
            if r + g > br:
                continue
            if r + g + 255 < br:
                continue
            b = br - r - g
            ret.append((r, g, b))
    return ret


if __name__ == '__main__':
    # List combinations of every brightness level
    color_cnt = 0
    for br in range(0, 255 * 3 + 1):
        colors = colors_of_brightness(br)
        color_cnt += len(colors)
        print(br, len(colors))

    print(color_cnt)
