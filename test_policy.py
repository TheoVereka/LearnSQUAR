from policy import *
myPo = Policy()
#print(exp(myPo.predict(myPo.params, Qubit(theta=pi,phi=0))))
for epoch in range(len(myPo.batch_size)):
    myPo.batch_update_params(epoch)
