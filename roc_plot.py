import matplotlib.pyplot as plt

data = [ (0.95,1), 
         (0.85,1),
         (0.8, -1), 
         (0.7, 1),
         (0.55, 1),
         (0.45,-1),
         (0.4,1),
         (0.3,1),
         (0.2,-1),
         (0.1,-1)]

n = len(data)

x_axis = []
y_axis = []
number_neg = 4
number_pos = n-number_neg
count= 0
TP = 0
FP = 0
last_p = 0
for i in range(len(data)):
    c,l = data[i]
    c_last,l_last = data[i-1]
    if i > 0 and c != c_last and l== -1 and TP > last_p:
        x_axis.append(FP/number_neg)
        y_axis.append(TP/number_pos)
        last_p = TP
        print(c,TP,FP)
    
    if l == 1:
        TP +=1
    else:
        FP+=1

x_axis.append(FP/number_neg)
y_axis.append(TP/number_pos)

print(x_axis,y_axis)
plt.plot(x_axis,y_axis)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_Q5.pdf")