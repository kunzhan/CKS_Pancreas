import re
import ipdb
import matplotlib.pyplot as plt
import os.path as osp
 
filepath = "/home/datasets/Pancreas82NIH/logs/FD0:Z3_1_20211101_191714.txt"


txt = open(filepath, "r").read()
 
result=""
test_text = re.findall("Loss+(......................)", txt)
result = result +'\n'.join(test_text)

result = result.replace(" ", "")
#result = result.replace("/","")
#print(result)

loss=""
avg_loss = re.findall("(......)+,",result)
loss = loss + '\n'.join(avg_loss)
#print(loss)
mode = {'Loss'}
count = 0
Loss = []
x = []
i = 0

with open('1.txt','w') as f:
	f.write(loss)

with open("1.txt", 'r') as f1:
	while True:
		#ipdb.set_trace()
		line = f1.readline().replace("\n","")

		if line == '':
			break
		#line = line.replace("\n","")
		count += 1
		if mode == {'Loss'}:
			Loss.append(float(line))
		x.append(count)

#ipdb.set_trace()
if mode == {'Loss'}:
	plt.plot(x, Loss)

plt.savefig('Z3_1_20211101_191714')
plt.show()