import matplotlib.pyplot as plt


# this function is to find the mode of the list
def mode(li):
    counts = {}
    for element in li:
        if element not in counts.keys():
            counts[element] = 1
        else:
            counts[element] = counts[element] + 1

    temp_list = []
    for key in counts.keys():
        temp_list.append((key, counts[key]))

    temp_list.sort(key=lambda x: x[1])

    return temp_list[-1][0]


# use to count the number of one element in a list
def count(li, x):
    number = 0
    for y in li:
        if y == x:
            number += 1
    return number


def draw_path(position_list, title, axis):
    # prepare parameters for plot function
    n = len(position_list)
    x_axis = [x for x in range(n)]
    y_axis = position_list

    # format line
    plt.plot(x_axis, y_axis, 'ro', color='b', marker='o', markersize=1)

    # set the boundary for better view
    if axis == 'left' or axis == 'right':
        plt.axis([0, n, -50, 640])
    elif axis == 'top' or axis == 'bottom':
        plt.axis([0, n, -50, 480])

    # set title and show
    plt.title('face_id ' + str(title) + ' ' + axis)
    plt.show()


def show_list(li):
    n = len(li)

    for i in range(n):
        print(li[i], end=';   ')

    print('\n')


if __name__ == '__main__':
    print(count(['dien', 'danh', 'dien', 'dien', 'dien', 'danh', 'danh', 'dien'], 'dien'))



