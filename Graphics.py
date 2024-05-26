import matplotlib.pyplot as plt

x = [1000, 2500, 5000, 7500, 10000]
Data = [1.1942262649536133, 2.3203251361846924, 3.5445680618286133,
        5.636908054351807, 6.399084091186523]

plt.plot(x, Data, label ="CFRAC", color ='#5451B6', linewidth = 1.5, marker ='.')

#plt.plot(x, simple_collision, label ="simple hash", color ='#C996FF', linewidth = 1.5, marker ='.')
#plt.plot(x, complex_collision, label ="complex hash", color ='#FF96CC', linewidth = 1.5, marker ='*')
plt.legend()
plt.show()
