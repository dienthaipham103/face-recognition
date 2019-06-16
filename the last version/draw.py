import matplotlib.pyplot as plt


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



