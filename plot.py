import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    graph_data = open('loss.txt', 'r').read()
    lines = graph_data.split('\n')
    arr = []
    ind = []
    cnt = 0
    for line in lines:
        if len(line) > 1:
            cnt = cnt + 1
            val = float(line)
            arr.append(val)
            ind.append(cnt)
    ax1.clear()
    ax1.plot(ind, arr)

ani = animation.FuncAnimation(fig, animate, interval = 100)
plt.show()