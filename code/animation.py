
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# rescale ticks


def load_data():
	print("\nLoading Data\n")
	for i in range(100):
		F1 = np.loadtxt('./FTLE_files/FTLE%d.txt'%i) 
		FTLE1.append(F1)
		F2 = np.loadtxt('./FTLE_files_model/FTLE%d.txt'%i) 
		FTLE2.append(F2)
	print("Loading Complete\n")

# initialize the data arrays 
def img_gen():
	t = 0
	while t < 99:
		F1 = FTLE1[t]
		F2 = FTLE2[t]
		# adapted the data generator to yield both sin and cos
		t+=1
		yield t,F1, F2

def run(data):
	# update the data
	t,F1, F2 = data

	# axis limits checking. Same as before, just for both axes
	for ax in [ax1, ax2]:
			ax.figure.canvas.draw()

	# update the data of both ims objects
	ims[0].set_data(F1)
	ims[1].set_data(F2)

	return ims

x = [0,100,200,300,395]
y = [0,100,195]
xlabels = ['0','0.5','1.0','1.5','2.0']
ylabels = ['1.0','0.5','0']

FTLE1=[]
FTLE2=[]
load_data()
# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1,2)

# intialize two ims objects (one in each axes)
ims1 = ax1.imshow(FTLE1[0],cmap='jet', animated=True)
ims2 = ax2.imshow(FTLE2[0],cmap='jet', animated=True)
ims = [ims1, ims2]

ax1.set_title("Data")
ax2.set_title("Model")
for ax in [ax1, ax2]:
	ax.set_xticks(x)
	ax.set_yticks(y)
	ax.set_xticklabels(xlabels)
	ax.set_yticklabels(ylabels)
	ax.grid()


ani = animation.FuncAnimation(fig, run, img_gen, interval=10, blit=True,repeat_delay=0)
ani.save('./FTLE_demo.gif', writer='imagemagick', fps=60)								
plt.show()